import os
import json
import re
import secrets
import ast
import random
import pandas as pd
import numpy as np
import boto3
from datetime import datetime
from io import BytesIO
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from urllib.parse import quote_plus
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from werkzeug.security import generate_password_hash, check_password_hash


# ===================================================================
# CONFIGURATION
# ===================================================================
GOOGLE_CLIENT_ID = "631387090392-32t9oqn1ph8sjsks4o57qe3225ks8rma.apps.googleusercontent.com"
load_dotenv()

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    secrets.token_hex(32)
)

# ================================================================
# SESSION CONFIGURATION
# ================================================================

app.config["SESSION_COOKIE_NAME"] = "skyguide_session"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False
app.config["SESSION_PERMANENT"] = False
CORS(
    app,
    supports_credentials=True,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:5500",
                "http://127.0.0.1:5500",
                "null"
            ]
        }
    }
)


# ===================================================================
# GROQ
# ===================================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add GROQ_API_KEY=your_key_here to .env"
    )

client = Groq(api_key=GROQ_API_KEY)


# ===================================================================
# FILE PATHS
# ===================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HOTEL_CSV = os.path.join(BASE_DIR, "hotels_dataset.csv")
TRAVEL_CSV = os.path.join(BASE_DIR, "travel_data.csv")
FLIGHT_CSV = os.path.join(BASE_DIR, "Clean_Dataset.csv")
TRAIN_SCHEDULE_CSV = os.path.join(BASE_DIR, "schedules.csv")
TRAIN_FARE_CSV = os.path.join(BASE_DIR, "price_data.csv.gz")

USERS_FILE = os.path.join(BASE_DIR, "users.json")
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")


# ===================================================================
# AWS S3
# LOCAL DATASET IS ALWAYS TRIED FIRST
# ===================================================================

S3_BUCKET = "travel-planner-67"

S3_HOTEL_KEY = "hotels/hotels_dataset.csv"
S3_FLIGHT_KEY = "transport/Clean_Dataset.csv"
S3_TRAIN_SCHEDULE_KEY = "transport/schedules.csv"
S3_TRAIN_FARE_KEY = "transport/price_data.csv"


# ===================================================================
# GROQ / DATAFRAMES
# ===================================================================

hotels_df = pd.DataFrame()
travel_df = pd.DataFrame()
flights_df = pd.DataFrame()
train_schedule_df = pd.DataFrame()
train_fares_df = pd.DataFrame()


# ===================================================================
# COMMON HELPERS
# ===================================================================

def clean_place(value):
    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip()
    ).lower()


def normalize_city(value):
    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value or "").lower()
    ).strip()

    aliases = {
        "bhubaneshwar": "bhubaneswar",
        "bhubaneswar": "bhubaneswar",
        "bengaluru": "bengaluru",
        "bangalore": "bengaluru",
        "bombay": "mumbai",
        "mumbai": "mumbai",
        "calcutta": "kolkata",
        "kolkata": "kolkata",
        "allahabad": "allahabad",
        "prayagraj": "allahabad",
        "trivandrum": "trivandrum",
        "thiruvananthapuram": "trivandrum",
    }

    return aliases.get(value, value)


def same_place(from_place, to_place):
    """
    Prevent source and destination from being identical.
    Also handles common city aliases.
    """
    a = normalize_city(from_place)
    b = normalize_city(to_place)

    if not a or not b:
        return False

    return a == b


# ===================================================================
# LOAD LOCAL FILE FIRST, S3 ONLY AS FALLBACK
# ===================================================================

def load_csv_local_first(
    local_path,
    s3_key,
    required_columns=None,
    numeric_columns=None,
    dataset_name="dataset"
):
    """
    LOCAL/OFFLINE FILE IS ALWAYS USED FIRST.

    Only if the local file doesn't exist or cannot be loaded
    correctly will the function try Amazon S3.
    """

    required_columns = required_columns or []
    numeric_columns = numeric_columns or []

    # ---------------------------------------------------------------
    # 1. LOCAL FILE FIRST
    # ---------------------------------------------------------------

    if os.path.exists(local_path):
        try:
            print(
                f"[info] Loading {dataset_name} from LOCAL dataset..."
            )

            df = pd.read_csv(local_path)
            df.columns = df.columns.str.strip()

            missing = set(required_columns) - set(df.columns)

            if not missing:

                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(
                            df[col],
                            errors="coerce"
                        )

                print(
                    f"[info] Loaded {len(df)} {dataset_name} "
                    f"records from LOCAL dataset."
                )

                return df

            print(
                f"[warn] Local {dataset_name} missing columns: "
                f"{missing}"
            )

        except Exception as e:
            print(
                f"[warn] Could not load local {dataset_name}: {e}"
            )

    else:
        print(
            f"[info] Local {dataset_name} not found."
        )

    # ---------------------------------------------------------------
    # 2. S3 FALLBACK
    # ---------------------------------------------------------------

    try:
        print(
            f"[info] Trying S3 fallback for {dataset_name}..."
        )

        s3 = boto3.client("s3")

        response = s3.get_object(
            Bucket=S3_BUCKET,
            Key=s3_key
        )

        df = pd.read_csv(
            BytesIO(response["Body"].read())
        )

        df.columns = df.columns.str.strip()

        missing = set(required_columns) - set(df.columns)

        if missing:
            print(
                f"[warn] S3 {dataset_name} missing columns: "
                f"{missing}"
            )
            return pd.DataFrame()

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        print(
            f"[info] Loaded {len(df)} {dataset_name} "
            f"records from S3 fallback."
        )

        return df

    except Exception as e:
        print(
            f"[warn] Could not load {dataset_name} from S3: {e}"
        )

        return pd.DataFrame()


# ===================================================================
# HOTEL DATASET
# ===================================================================

hotels_df = load_csv_local_first(
    HOTEL_CSV,
    S3_HOTEL_KEY,
    required_columns=[
        "Hotel_Name",
        "Hotel_Rating",
        "City",
        "Hotel_Price"
    ],
    numeric_columns=[
        "Hotel_Rating",
        "Hotel_Price"
    ],
    dataset_name="hotel"
)

if not hotels_df.empty:

    hotels_df["City"] = (
        hotels_df["City"]
        .astype(str)
        .str.strip()
    )

    hotels_df = hotels_df.dropna(
        subset=[
            "Hotel_Name",
            "City",
            "Hotel_Price"
        ]
    )

    print(
        f"[info] Hotels ready: {len(hotels_df)} records"
    )


# ===================================================================
# LOCAL DESTINATION IMAGES
# ===================================================================

IMAGE_MAP = {
    "Mumbai": "images/mumbai.webp",
    "Goa": "images/goa.webp",
    "Jaipur": "images/jaipur.webp",
    "Ladakh": "images/ladakh.webp",
    "Kolkata": "images/kolkata.webp",
    "Jodhpur": "images/jodhpur.webp",
    "Varanasi": "images/varanasi.webp",
    "Delhi": "images/delhi.webp",
    "Amritsar": "images/amritsar.webp",
}


REMOTE_IMAGE_FALLBACKS = {
    "Mumbai":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/Mumbai_Aug_2018_ID.jpg/1200px-Mumbai_Aug_2018_ID.jpg",

    "Goa":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Goa_Palolem_Beach.jpg/1200px-Goa_Palolem_Beach.jpg",

    "Jaipur":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Hawa_Mahal_2011.jpg/1200px-Hawa_Mahal_2011.jpg",

    "Ladakh":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Pangong_Lake_Ladakh_India.jpg/1200px-Pangong_Lake_Ladakh_India.jpg",

    "Kolkata":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Howrah_bridge_kolkata.jpg/1200px-Howrah_bridge_kolkata.jpg",

    "Jodhpur":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Jodhpur_the_blue_city.JPG/1200px-Jodhpur_the_blue_city.JPG",

    "Varanasi":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Varanasi_Ghat.jpg/1200px-Varanasi_Ghat.jpg",

    "Delhi":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/India_Gate_in_New_Delhi_03-2016.jpg/1200px-India_Gate_in_New_Delhi_03-2016.jpg",

    "Amritsar":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/The_Golden_Temple.jpg/1200px-The_Golden_Temple.jpg",
}


# ===================================================================
# TRANSPORT DISTRICTS / STATES
# ===================================================================

DISTRICT_STATES = {
    "Visakhapatnam": "Andhra Pradesh",
    "Vijayawada": "Andhra Pradesh",
    "Bhubaneswar": "Odisha",
    "Hyderabad": "Telangana",
    "Bengaluru Urban": "Karnataka",
    "Chennai": "Tamil Nadu",
    "Mumbai City": "Maharashtra",
    "Pune": "Maharashtra",
    "Kolkata": "West Bengal",
    "Jaipur": "Rajasthan",
    "Amritsar": "Punjab",
    "Ahmedabad": "Gujarat",
    "Bhopal": "Madhya Pradesh",
    "Chandigarh": "Chandigarh",
    "Coimbatore": "Tamil Nadu",
    "Dehradun": "Uttarakhand",
    "Delhi": "Delhi",
    "Goa": "Goa",
    "Guwahati": "Assam",
    "Indore": "Madhya Pradesh",
    "Kanpur Nagar": "Uttar Pradesh",
    "Kochi": "Kerala",
    "Lucknow": "Uttar Pradesh",
    "Nagpur": "Maharashtra",
    "Patna": "Bihar",
    "Ranchi": "Jharkhand",
    "Surat": "Gujarat",
    "Thane": "Maharashtra",
    "Varanasi": "Uttar Pradesh",
}


DISTRICT_ALIASES = {
    "Visakhapatnam": [
        "visakhapatnam",
        "vizag",
        "visakhapatnam airport",
        "vishakhapatnam"
    ],
    "Vijayawada": [
        "vijayawada",
        "bezawada"
    ],
    "Bhubaneswar": [
        "bhubaneswar",
        "bhubaneshwar"
    ],
    "Hyderabad": [
        "hyderabad",
        "secunderabad"
    ],
    "Bengaluru Urban": [
        "bangalore",
        "bengaluru",
        "yesvantpur",
        "yelahanka"
    ],
    "Chennai": [
        "chennai",
        "madras"
    ],
    "Mumbai City": [
        "mumbai",
        "bombay",
        "csmt",
        "lokmanya tilak"
    ],
    "Pune": ["pune"],
    "Kolkata": [
        "kolkata",
        "calcutta",
        "howrah",
        "sealdah"
    ],
    "Jaipur": ["jaipur"],
    "Amritsar": ["amritsar"],
    "Ahmedabad": [
        "ahmedabad",
        "sabarmati"
    ],
    "Bhopal": ["bhopal"],
    "Chandigarh": ["chandigarh"],
    "Coimbatore": ["coimbatore"],
    "Dehradun": ["dehradun"],
    "Delhi": [
        "delhi",
        "new delhi",
        "anand vihar",
        "hazrat nizamuddin"
    ],
    "Goa": [
        "goa",
        "madgaon",
        "madgaon junction",
        "vasco da gama"
    ],
    "Guwahati": ["guwahati"],
    "Indore": ["indore"],
    "Kanpur Nagar": ["kanpur"],
    "Kochi": [
        "kochi",
        "ernakulam"
    ],
    "Lucknow": ["lucknow"],
    "Nagpur": ["nagpur"],
    "Patna": ["patna"],
    "Ranchi": ["ranchi"],
    "Surat": ["surat"],
    "Thane": ["thane"],
    "Varanasi": [
        "varanasi",
        "banaras"
    ],
}


# ===================================================================
# TRANSPORT DESTINATIONS
# ===================================================================

def build_transport_destinations():
    """
    Build the transport destination list independently
    from the homepage featured-location list.

    Transport should show ALL Indian destinations available
    in the travel/hotel datasets.
    """

    destinations = set()

    # ---------------------------------------------------------------
    # 1. Destinations from travel_data.csv
    # ---------------------------------------------------------------

    if (
        not travel_df.empty
        and "city" in travel_df.columns
    ):

        for city in travel_df["city"].dropna():

            city = str(city).strip()

            if city:
                destinations.add(city.title())

    # ---------------------------------------------------------------
    # 2. Also include cities from hotel dataset
    # ---------------------------------------------------------------

    if (
        not hotels_df.empty
        and "City" in hotels_df.columns
    ):

        for city in hotels_df["City"].dropna():

            city = str(city).strip()

            if city:
                destinations.add(city.title())

    # ---------------------------------------------------------------
    # Remove invalid / empty values
    # ---------------------------------------------------------------

    destinations = {
        city
        for city in destinations
        if city
        and city.lower() not in {
            "nan",
            "none",
            "null"
        }
    }

    return sorted(
        destinations,
        key=lambda x: x.lower()
    )


TRANSPORT_DESTINATIONS = build_transport_destinations()


# ===================================================================
# DESTINATION STATES
# ===================================================================

DESTINATION_STATES = {
    "Goa": "Goa",
    "Kolkata": "West Bengal",
    "Mumbai": "Maharashtra",
    "Jaipur": "Rajasthan",
    "Amritsar": "Punjab",

    "Delhi": "Delhi",
    "Varanasi": "Uttar Pradesh",
    "Jodhpur": "Rajasthan",
    "Ladakh": "Ladakh",

    "Visakhapatnam": "Andhra Pradesh",
    "Vijayawada": "Andhra Pradesh",
    "Bhubaneswar": "Odisha",
    "Hyderabad": "Telangana",
    "Bengaluru": "Karnataka",
    "Bangalore": "Karnataka",
    "Chennai": "Tamil Nadu",
    "Pune": "Maharashtra",
    "Ahmedabad": "Gujarat",
    "Bhopal": "Madhya Pradesh",
    "Chandigarh": "Chandigarh",
    "Coimbatore": "Tamil Nadu",
    "Dehradun": "Uttarakhand",
    "Guwahati": "Assam",
    "Indore": "Madhya Pradesh",
    "Kanpur": "Uttar Pradesh",
    "Kochi": "Kerala",
    "Lucknow": "Uttar Pradesh",
    "Nagpur": "Maharashtra",
    "Patna": "Bihar",
    "Ranchi": "Jharkhand",
    "Surat": "Gujarat",
    "Thane": "Maharashtra",
}


def get_destination_state(destination):
    """
    Return the known state for a destination.

    If the city is not in the manually maintained mapping,
    return 'Other' instead of blocking the destination.
    """

    destination_normalized = normalize_city(
        destination
    )

    for city, state in DESTINATION_STATES.items():

        if normalize_city(city) == destination_normalized:
            return state

    return "Other"

AIRLINES = [
    "Air India",
    "IndiGo",
    "SpiceJet",
    "Akasa Air",
    "Air India Express"
]


# ===================================================================
# HOTEL BUDGET TIERS
# ===================================================================

# Simplified tiers for users.
BUDGET_TIERS = {
    "Budget": (0, 1999),
    "Standard": (2000, 4999),
    "Premium": (5000, 9999),
    "Luxury": (10000, float("inf")),
}


def get_tier(price):
    try:
        price = float(price)
    except (TypeError, ValueError):
        return "Standard"

    for tier, (low, high) in BUDGET_TIERS.items():
        if low <= price <= high:
            return tier

    return "Luxury"


# ===================================================================
# TRAVEL DATASET
# ===================================================================

def load_travel_data():

    if not os.path.exists(TRAVEL_CSV):
        return pd.DataFrame()

    try:
        df = pd.read_csv(TRAVEL_CSV)
        df.columns = df.columns.str.strip()

        if "country" in df.columns and "city" in df.columns:
            return df[
                df["country"]
                .astype(str)
                .str.contains(
                    "India",
                    case=False,
                    na=False
                )
            ].copy()

    except Exception as e:
        print(
            f"[warn] Could not read travel_data.csv: {e}"
        )

    return pd.DataFrame()


travel_df = load_travel_data()


# ===================================================================
# TRAVEL THEME HELPERS
# ===================================================================

def top_themes(row, n=3):

    theme_cols = [
        "culture",
        "adventure",
        "nature",
        "beaches",
        "nightlife",
        "cuisine",
        "wellness",
        "urban",
        "seclusion"
    ]

    available = {
        c: row.get(c, 0)
        for c in theme_cols
        if c in row.index
    }

    if available:
        return sorted(
            available,
            key=available.get,
            reverse=True
        )[:n]

    return [
        "hotels",
        "exploration",
        "travel"
    ]


# ===================================================================
# CITY / HOTEL MATCHING
# ===================================================================

def city_match(city):

    if hotels_df.empty:
        return hotels_df

    requested = normalize_city(city)

    normalized = hotels_df[
        "City"
    ].map(normalize_city)

    exact = hotels_df[
        normalized == requested
    ]

    if not exact.empty:
        return exact

    return hotels_df[
        normalized.str.contains(
            re.escape(requested),
            na=False
        )
    ]


def normalize_hotel_name(name):
    """
    Normalizes hotel names so duplicates caused by
    spaces/capitalization/minor formatting are removed.
    """

    value = str(name or "").lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


def remove_duplicate_hotels(df):
    """
    Remove duplicate hotel records.

    If the same hotel appears multiple times, keep:
    1. Highest rating
    2. Lowest price
    """

    if df.empty or "Hotel_Name" not in df.columns:
        return df

    result = df.copy()

    result["_normalized_hotel_name"] = (
        result["Hotel_Name"]
        .map(normalize_hotel_name)
    )

    result = result.sort_values(
        [
            "_normalized_hotel_name",
            "Hotel_Rating",
            "Hotel_Price"
        ],
        ascending=[
            True,
            False,
            True
        ]
    )

    result = result.drop_duplicates(
        subset=["_normalized_hotel_name"],
        keep="first"
    )

    result = result.drop(
        columns=["_normalized_hotel_name"],
        errors="ignore"
    )

    return result


# ===================================================================
# TRANSPORT DATASETS
# ===================================================================

flights_df = load_csv_local_first(
    FLIGHT_CSV,
    S3_FLIGHT_KEY,
    required_columns=[
        "source_city",
        "destination_city",
        "class",
        "price"
    ],
    numeric_columns=[
        "price"
    ],
    dataset_name="flight"
)


if not flights_df.empty:

    flights_df["source_city"] = (
        flights_df["source_city"]
        .astype(str)
        .str.strip()
    )

    flights_df["destination_city"] = (
        flights_df["destination_city"]
        .astype(str)
        .str.strip()
    )

    flights_df["class"] = (
        flights_df["class"]
        .astype(str)
        .str.strip()
    )

    flights_df = flights_df.dropna(
        subset=[
            "source_city",
            "destination_city",
            "price"
        ]
    )


# ===================================================================
# TRAIN SCHEDULE
# ===================================================================

train_schedule_df = load_csv_local_first(
    TRAIN_SCHEDULE_CSV,
    S3_TRAIN_SCHEDULE_KEY,
    dataset_name="train schedule"
)


# ===================================================================
# TRAIN FARES
# ===================================================================

train_fares_df = load_csv_local_first(
    TRAIN_FARE_CSV,
    S3_TRAIN_FARE_KEY,
    numeric_columns=[
        "totalFare",
        "distance",
        "trainNumber"
    ],
    dataset_name="train fare"
)


# ===================================================================
# TRAIN HELPERS
# ===================================================================

def parse_station_list(raw):

    try:
        value = json.loads(
            str(raw).replace("'", '"')
        )

    except Exception:

        try:
            value = ast.literal_eval(
                str(raw)
            )

        except Exception:
            return []

    return value if isinstance(value, list) else []


def station_matches(station, query):

    q = clean_place(query)

    if not q:
        return False

    code = clean_place(
        station.get("stationCode")
    )

    name = clean_place(
        station.get("stationName")
    )

    return (
        q == code or
        q == name or
        q in name or
        q in code
    )


def station_catalog():

    seen = {}

    if (
        train_schedule_df.empty or
        "stationList" not in train_schedule_df.columns
    ):
        return []

    for raw in train_schedule_df[
        "stationList"
    ].dropna():

        for station in parse_station_list(
            str(raw)
        ):

            code = str(
                station.get("stationCode", "")
            ).strip()

            name = str(
                station.get("stationName", "")
            ).strip()

            if code and name:
                seen[code] = name

    return [
        {
            "code": code,
            "name": name
        }
        for code, name in sorted(
            seen.items(),
            key=lambda x: x[1].lower()
        )
    ]


CITY_STATION_ALIASES = {
    "vizag": [
        "VSKP",
        "VISAKHAPATNAM"
    ],
    "visakhapatnam": [
        "VSKP",
        "VISAKHAPATNAM"
    ],
    "goa": [
        "MAO",
        "MADGAON",
        "VSG",
        "VASCO DA GAMA"
    ],
    "mumbai": [
        "CSMT",
        "LTT",
        "DR",
        "BDTS",
        "BCT"
    ],
    "delhi": [
        "NDLS",
        "DLI",
        "ANVT",
        "NZM"
    ],
    "kolkata": [
        "HWH",
        "KOAA",
        "SDAH"
    ],
    "jaipur": ["JP"],
    "amritsar": ["ASR"],
}


def station_query_matches(station, query):

    q = clean_place(query)

    if station_matches(station, q):
        return True

    aliases = CITY_STATION_ALIASES.get(
        q,
        []
    )

    code = clean_place(
        station.get("stationCode")
    )

    name = clean_place(
        station.get("stationName")
    )

    for alias in aliases:

        a = clean_place(alias)

        if (
            code == a or
            a in name
        ):
            return True

    return False


def minutes_from_time(
    value,
    day_count=1
):

    if not value or value == "--":
        return None

    try:

        hh, mm = [
            int(x)
            for x in str(value).split(":")[:2]
        ]

        return (
            (int(day_count or 1) - 1)
            * 1440
            + hh * 60
            + mm
        )

    except Exception:
        return None


# ===================================================================
# TRAIN FARE ESTIMATION
# ===================================================================

def estimate_train_fare(
    train_number,
    from_code,
    to_code,
    distance,
    class_code
):

    if train_fares_df.empty:
        return None, "unavailable"

    fares = train_fares_df.copy()

    fares["trainNumber"] = pd.to_numeric(
        fares["trainNumber"],
        errors="coerce"
    )

    exact = fares[
        (fares["trainNumber"] ==
         float(train_number))
        &
        (
            fares["fromStnCode"]
            .astype(str)
            .str.upper()
            ==
            str(from_code).upper()
        )
        &
        (
            fares["toStnCode"]
            .astype(str)
            .str.upper()
            ==
            str(to_code).upper()
        )
        &
        (
            fares["classCode"]
            .astype(str)
            .str.upper()
            ==
            str(class_code).upper()
        )
    ]

    if not exact.empty:

        fare = (
            exact["totalFare"]
            .dropna()
        )

        if not fare.empty:

            return (
                round(
                    float(fare.median()),
                    2
                ),
                "dataset exact segment"
            )

    same_train = fares[
        (fares["trainNumber"] ==
         float(train_number))
        &
        (
            fares["classCode"]
            .astype(str)
            .str.upper()
            ==
            str(class_code).upper()
        )
        &
        (fares["distance"] > 0)
        &
        (fares["totalFare"] > 0)
    ].copy()

    if not same_train.empty and distance:

        rate = (
            same_train["totalFare"]
            /
            same_train["distance"]
        )

        rate = (
            rate
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if not rate.empty:

            return (
                round(
                    float(rate.median() * distance),
                    2
                ),
                "dataset distance estimate"
            )

    same_class = fares[
        (
            fares["classCode"]
            .astype(str)
            .str.upper()
            ==
            str(class_code).upper()
        )
        &
        (fares["distance"] > 0)
        &
        (fares["totalFare"] > 0)
    ].copy()

    if not same_class.empty and distance:

        rate = (
            same_class["totalFare"]
            /
            same_class["distance"]
        )

        rate = (
            rate
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if not rate.empty:

            return (
                round(
                    float(rate.median() * distance),
                    2
                ),
                "dataset class estimate"
            )

    return None, "unavailable"


# ===================================================================
# FIND TRAIN ROUTES
# ===================================================================

def find_train_routes(
    from_query,
    to_query,
    class_code="SL",
    limit=12
):

    if train_schedule_df.empty:
        return []

    if same_place(
        from_query,
        to_query
    ):
        return []

    results = []

    for _, row in train_schedule_df.iterrows():

        stations = parse_station_list(
            str(
                row.get(
                    "stationList",
                    ""
                )
            )
        )

        if not stations:
            continue

        from_idx = None
        to_idx = None

        for i, station in enumerate(stations):

            if (
                from_idx is None and
                station_query_matches(
                    station,
                    from_query
                )
            ):
                from_idx = i
                continue

            if (
                from_idx is not None and
                station_query_matches(
                    station,
                    to_query
                )
            ):
                to_idx = i
                break

        if (
            from_idx is None or
            to_idx is None or
            to_idx <= from_idx
        ):
            continue

        a = stations[from_idx]
        b = stations[to_idx]

        try:

            distance_a = float(
                a.get("distance", 0) or 0
            )

            distance_b = float(
                b.get("distance", 0) or 0
            )

            distance = max(
                0,
                distance_b - distance_a
            )

        except Exception:
            distance = 0

        start = minutes_from_time(
            a.get("departureTime")
            if a.get("departureTime") != "--"
            else a.get("arrivalTime"),
            a.get("dayCount", 1)
        )

        end = minutes_from_time(
            b.get("arrivalTime")
            if b.get("arrivalTime") != "--"
            else b.get("departureTime"),
            b.get("dayCount", 1)
        )

        duration_minutes = None

        if (
            start is not None and
            end is not None
        ):
            duration_minutes = max(
                0,
                end - start
            )

        train_number = row.get(
            "trainNumber"
        )

        fare, fare_source = estimate_train_fare(
            train_number,
            a.get("stationCode"),
            b.get("stationCode"),
            distance,
            class_code
        )

        result = {
            "train_number": str(
                train_number
            ),
            "train_name": str(
                row.get(
                    "trainName",
                    "Indian Railways"
                )
            ),
            "from": str(
                a.get(
                    "stationName",
                    a.get("stationCode", "")
                )
            ).title(),
            "from_code": str(
                a.get("stationCode", "")
            ),
            "to": str(
                b.get(
                    "stationName",
                    b.get("stationCode", "")
                )
            ).title(),
            "to_code": str(
                b.get("stationCode", "")
            ),
            "class_code": class_code.upper(),
            "distance_km": round(
                distance,
                1
            ),
            "duration_minutes": duration_minutes,
            "fare_per_person": fare,
            "fare_source": fare_source,
            "runs": {
                "Mon": row.get(
                    "trainRunsOnMon"
                ) == "Y",
                "Tue": row.get(
                    "trainRunsOnTue"
                ) == "Y",
                "Wed": row.get(
                    "trainRunsOnWed"
                ) == "Y",
                "Thu": row.get(
                    "trainRunsOnThu"
                ) == "Y",
                "Fri": row.get(
                    "trainRunsOnFri"
                ) == "Y",
                "Sat": row.get(
                    "trainRunsOnSat"
                ) == "Y",
                "Sun": row.get(
                    "trainRunsOnSun"
                ) == "Y",
            }
        }

        results.append(result)

    results.sort(
        key=lambda x: (
            x["fare_per_person"] is None,
            x["duration_minutes"] is None,
            x["duration_minutes"] or 999999
        )
    )

    return results[:limit]


# ===================================================================
# FLIGHTS
# ===================================================================

def flight_options(
    source,
    destination,
    class_name="Economy",
    limit=12
):

    if flights_df.empty:
        return []

    if same_place(
        source,
        destination
    ):
        return []

    src = clean_place(source)
    dst = clean_place(destination)

    matches = flights_df[
        flights_df["source_city"]
        .str.lower()
        .eq(src)
        &
        flights_df["destination_city"]
        .str.lower()
        .eq(dst)
    ]

    if matches.empty:

        matches = flights_df[
            flights_df["source_city"]
            .str.lower()
            .str.contains(
                re.escape(src),
                na=False
            )
            &
            flights_df["destination_city"]
            .str.lower()
            .str.contains(
                re.escape(dst),
                na=False
            )
        ]

    if class_name:

        class_matches = matches[
            matches["class"]
            .str.lower()
            ==
            clean_place(class_name)
        ]

        if not class_matches.empty:
            matches = class_matches

    if matches.empty:
        return []

    columns = [
        "airline",
        "flight",
        "source_city",
        "destination_city",
        "departure_time",
        "arrival_time",
        "stops",
        "class",
        "duration",
        "days_left",
        "price"
    ]

    columns = [
        c for c in columns
        if c in matches.columns
    ]

    return (
        matches
        .sort_values("price")
        .head(limit)
        [columns]
        .to_dict(orient="records")
    )


# ===================================================================
# TRANSPORT FALLBACK
# ===================================================================

def district_state(place):

    key = str(place or "").strip()

    if key in DISTRICT_STATES:
        return DISTRICT_STATES[key]

    n = normalize_city(key)

    for district, aliases in DISTRICT_ALIASES.items():

        if (
            n == normalize_city(district)
            or
            any(
                n == normalize_city(alias)
                for alias in aliases
            )
        ):
            return DISTRICT_STATES.get(
                district,
                "Other"
            )

    return "Other"


def fallback_transport_options(
    from_place,
    to_place,
    mode,
    class_name,
    travelers=1
):

    if same_place(
        from_place,
        to_place
    ):
        return []

    from_state = district_state(
        from_place
    )

    to_state = get_destination_state(
    to_place
    )

    same_state = (
        from_state != "Other"
        and
        from_state == to_state
    )

    mode_key = clean_place(mode)
    class_key = clean_place(class_name)

    if mode_key == "flight":

        if class_key == "business":

            low, high = (
                (4500, 7000)
                if same_state
                else
                (8000, 10000)
            )

        else:

            low, high = (
                (2000, 4500)
                if same_state
                else
                (4500, 8000)
            )

        return [{
            "airline": random.choice(
                AIRLINES
            ),
            "flight": f"AI-{random.randint(100, 999)}",
            "source_city": from_place,
            "destination_city": to_place,
            "departure_time": "Estimated",
            "arrival_time": "Estimated",
            "stops": "Non-stop / estimated",
            "class": class_name,
            "duration": (
                2.0
                if same_state
                else
                3.0
            ),
            "days_left": 0,
            "price": random.randint(
                low,
                high
            ),
            "estimated": True
        }]

    if mode_key == "train":

        if class_key in {
            "1a",
            "first ac"
        }:

            low, high = (
                (1200, 2500)
                if same_state
                else
                (2500, 5000)
            )

        elif class_key in {
            "2a",
            "ac 2 tier"
        }:

            low, high = (
                (700, 1600)
                if same_state
                else
                (1500, 3200)
            )

        elif class_key in {
            "3a",
            "ac 3 tier"
        }:

            low, high = (
                (450, 1100)
                if same_state
                else
                (900, 2200)
            )

        else:

            low, high = (
                (250, 700)
                if same_state
                else
                (500, 1600)
            )

        return [{
            "train_number": str(
                random.randint(
                    10000,
                    99999
                )
            ),
            "train_name":
                "Indian Railways · Estimated service",
            "from": from_place,
            "from_code": "",
            "to": to_place,
            "to_code": "",
            "class_code":
                class_name.upper(),
            "distance_km": None,
            "duration_minutes": None,
            "fare_per_person":
                random.randint(
                    low,
                    high
                ),
            "fare_source":
                "Estimated fare",
            "estimated": True
        }]

    if mode_key == "bus":

        low, high = (
            (300, 1500)
            if same_state
            else
            (700, 2500)
        )

        return [{
            "operator":
                "Intercity Bus · Estimated",
            "source_city":
                from_place,
            "destination_city":
                to_place,
            "class":
                class_name,
            "price":
                random.randint(
                    low,
                    high
                ),
            "estimated": True
        }]

    return []


# ===================================================================
# TRANSPORT FARE CALCULATION
# ===================================================================

def calculate_transport_fare(
    from_place,
    to_place,
    mode,
    class_name,
    travelers=1
):

    try:
        travelers = max(
            1,
            int(travelers)
        )
    except Exception:
        travelers = 1

    if same_place(
        from_place,
        to_place
    ):
        return (
            0,
            0,
            "Invalid: source and destination are the same"
        )

    mode_key = clean_place(mode)

    if mode_key == "flight":

        options = flight_options(
            from_place,
            to_place,
            class_name,
            limit=50
        )

        prices = [
            float(x["price"])
            for x in options
            if x.get("price") is not None
        ]

        if prices:

            per_person = round(
                float(
                    pd.Series(
                        prices
                    ).median()
                ),
                2
            )

            return (
                per_person,
                round(
                    per_person * travelers,
                    2
                ),
                "Flight dataset median"
            )

        options = fallback_transport_options(
            from_place,
            to_place,
            "Flight",
            class_name,
            travelers
        )

        if not options:
            return 0, 0, "Unavailable"

        per_person = float(
            options[0]["price"]
        )

        return (
            per_person,
            round(
                per_person * travelers,
                2
            ),
            f"Estimated · {options[0]['airline']}"
        )

    if mode_key == "train":

        trains = find_train_routes(
            from_place,
            to_place,
            class_name or "SL",
            limit=20
        )

        fares = [
            float(x["fare_per_person"])
            for x in trains
            if x.get("fare_per_person") is not None
        ]

        if fares:

            per_person = round(
                float(
                    pd.Series(
                        fares
                    ).median()
                ),
                2
            )

            return (
                per_person,
                round(
                    per_person * travelers,
                    2
                ),
                "Train fare dataset median"
            )

        options = fallback_transport_options(
            from_place,
            to_place,
            "Train",
            class_name or "SL",
            travelers
        )

        if not options:
            return 0, 0, "Unavailable"

        per_person = float(
            options[0]["fare_per_person"]
        )

        return (
            per_person,
            round(
                per_person * travelers,
                2
            ),
            "Estimated train fare"
        )

    if mode_key == "bus":

        options = fallback_transport_options(
            from_place,
            to_place,
            "Bus",
            class_name or "Standard",
            travelers
        )

        if options:

            per_person = float(
                options[0]["price"]
            )

            return (
                per_person,
                round(
                    per_person * travelers,
                    2
                ),
                "Estimated bus fare"
            )

    return 0, 0, "Not selected"

@app.route("/")
def home():
    return send_from_directory(".", "index.html")
# ===================================================================
# TRANSPORT LOCATIONS
# ===================================================================

@app.route(
    "/transport-locations",
    methods=["GET"]
)
def transport_locations():

    return jsonify({
        "districts":
            list(DISTRICT_STATES.keys()),
        "destinations":
            TRANSPORT_DESTINATIONS,
        "stations":
            station_catalog()
    })


# ===================================================================
# FLIGHT API
# ===================================================================

@app.route(
    "/transport/flights",
    methods=["GET"]
)
def transport_flights():

    from_place = request.args.get(
        "from",
        ""
    ).strip()

    to_place = request.args.get(
        "to",
        ""
    ).strip()

    class_name = request.args.get(
        "class",
        "Economy"
    ).strip()

    if (
        from_place not in DISTRICT_STATES
        or
        to_place not in TRANSPORT_DESTINATIONS
    ):

        return jsonify({
            "message":
                "Select a valid district and destination."
        }), 400

    # CHANGE #2
    if same_place(
        from_place,
        to_place
    ):

        return jsonify({
            "message":
                "Source and destination cannot be the same."
        }), 400

    results = flight_options(
        from_place,
        to_place,
        class_name
    )

    estimated = False

    if not results:

        results = fallback_transport_options(
            from_place,
            to_place,
            "Flight",
            class_name
        )

        estimated = True

    return jsonify({
        "mode": "Flight",
        "from": from_place,
        "to": to_place,
        "class": class_name,
        "count": len(results),
        "estimated": estimated,
        "flights": results
    })


# ===================================================================
# TRAIN API
# ===================================================================

@app.route(
    "/transport/trains",
    methods=["GET"]
)
def transport_trains():

    from_place = request.args.get(
        "from",
        ""
    ).strip()

    to_place = request.args.get(
        "to",
        ""
    ).strip()

    class_code = request.args.get(
        "class",
        "SL"
    ).strip().upper()

    if (
        from_place not in DISTRICT_STATES
        or
        to_place not in TRANSPORT_DESTINATIONS
    ):

        return jsonify({
            "message":
                "Select a valid district and destination."
        }), 400

    # CHANGE #2
    if same_place(
        from_place,
        to_place
    ):

        return jsonify({
            "message":
                "Source and destination cannot be the same."
        }), 400

    results = find_train_routes(
        from_place,
        to_place,
        class_code
    )

    estimated = False

    if (
        not results
        or
        not any(
            x.get("fare_per_person")
            is not None
            for x in results
        )
    ):

        results = fallback_transport_options(
            from_place,
            to_place,
            "Train",
            class_code
        )

        estimated = True

    return jsonify({
        "mode": "Train",
        "from": from_place,
        "to": to_place,
        "class": class_code,
        "count": len(results),
        "estimated": estimated,
        "trains": results
    })


# ===================================================================
# AUTHENTICATION
# ===================================================================

def load_users():

    if not os.path.exists(USERS_FILE):
        return {}

    try:

        with open(
            USERS_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    except Exception:
        return {}


def save_users(users):

    temp = USERS_FILE + ".tmp"

    with open(
        temp,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            users,
            f,
            indent=2
        )

    os.replace(
        temp,
        USERS_FILE
    )


@app.route(
    "/auth",
    methods=["POST"]
)
def auth():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    mode = (
        data.get("mode")
        or "login"
    ).lower()

    email = (
        data.get("email")
        or ""
    ).strip().lower()

    password = (
        data.get("password")
        or ""
    )

    name = (
        data.get("name")
        or ""
    ).strip()

    if not email or not password:

        return jsonify({
            "status": "error",
            "message":
                "Email and password are required."
        }), 400

    if not re.fullmatch(
        r"[^\s@]+@[^\s@]+\.[^\s@]+",
        email
    ):

        return jsonify({
            "status": "error",
            "message":
                "Please enter a valid email address."
        }), 400

    if len(password) < 6:

        return jsonify({
            "status": "error",
            "message":
                "Password must be at least 6 characters long."
        }), 400

    users = load_users()

    # ------------------------------------------------------------
    # REGISTER
    # ------------------------------------------------------------

    if mode == "register":

        if not name:

            return jsonify({
                "status": "error",
                "message":
                    "Please enter your full name."
            }), 400

        if email in users:

            return jsonify({
                "status": "error",
                "message":
                    "An account with this email already exists. "
                    "Please login."
            }), 409

        users[email] = {
            "name": name,
            "password":
                generate_password_hash(password)
        }

        save_users(users)

        return jsonify({
            "status": "registered",
            "message":
                "Account created successfully. "
                "Please login."
        })

    # ------------------------------------------------------------
    # LOGIN
    # ------------------------------------------------------------

    if email not in users:

        return jsonify({
            "status": "error",
            "message":
                "No account found with this email. "
                "Please register first."
        }), 401

    if not check_password_hash(
        users[email]["password"],
        password
    ):

        return jsonify({
            "status": "error",
            "message":
                "Incorrect email or password."
        }), 401

    # ------------------------------------------------------------
    # CREATE SESSION
    # ------------------------------------------------------------

    session.clear()

    session["user_email"] = email
    session["user_name"] = users[email]["name"]

    session.modified = True

    print(
        f"[auth] Login successful: {email}"
    )

    print(
        f"[auth] Session user_email: "
        f"{session.get('user_email')}"
    )

    return jsonify({
        "status": "success",
        "email": email,
        "name": users[email]["name"]
    })


@app.route(
    "/auth/status",
    methods=["GET"]
)
def auth_status():

    if "user_email" not in session:

        return jsonify({
            "logged_in": False
        })

    return jsonify({
        "logged_in": True,
        "email": session["user_email"],
        "name": session.get(
            "user_name"
        )
    })


@app.route(
    "/auth/logout",
    methods=["POST"]
)
def auth_logout():

    # Completely destroy the server-side session.
    session.clear()

    return jsonify({
        "status": "success",
        "logged_out": True
    })


# ===================================================================
# LOCAL IMAGES
# ===================================================================

@app.route(
    "/images/<path:filename>",
    methods=["GET"]
)
def local_image(filename):

    images_dir = os.path.join(
        BASE_DIR,
        "images"
    )

    return send_from_directory(
        images_dir,
        filename
    )


# ===================================================================
# DESTINATIONS
# ===================================================================

@app.route(
    "/get-popular",
    methods=["GET"]
)
def get_popular():

    records = []

    if (
        not travel_df.empty
        and
        "city" in travel_df.columns
    ):

        for _, row in travel_df.iterrows():

            city = str(
                row["city"]
            ).strip()

            records.append({
                "city": city,
                "country": "India",
                "region":
                    row.get(
                        "region",
                        "India"
                    ),
                "short_description":
                    row.get(
                        "short_description",
                        f"Explore {city} with SkyGuide."
                    ),
                "budget_level":
                    row.get(
                        "budget_level",
                        "Standard"
                    ),
                "top_themes":
                    top_themes(row),
                "img_url":
                    IMAGE_MAP.get(
                        city,
                        REMOTE_IMAGE_FALLBACKS.get(
                            city,
                            ""
                        )
                    ),
                "image_fallback":
                    REMOTE_IMAGE_FALLBACKS.get(
                        city,
                        ""
                    )
            })

    elif not hotels_df.empty:

        for city in sorted(
            hotels_df["City"]
            .unique()
        ):

            city_df = city_match(city)

            city_df = remove_duplicate_hotels(
                city_df
            )

            median_price = (
                city_df["Hotel_Price"]
                .median()
            )

            records.append({
                "city":
                    city.title(),
                "country":
                    "India",
                "region":
                    "India",
                "short_description":
                    f"Explore {city.title()} with SkyGuide. "
                    f"{len(city_df)} hotels are available "
                    f"in the dataset.",
                "budget_level":
                    get_tier(median_price),
                "top_themes": [
                    "hotels",
                    "exploration",
                    "travel"
                ],
                "img_url":
                    IMAGE_MAP.get(
                        city.title(),
                        ""
                    ),
                "image_fallback":
                    REMOTE_IMAGE_FALLBACKS.get(
                        city.title(),
                        ""
                    )
            })

    return jsonify(records)


@app.route(
    "/hotel-cities",
    methods=["GET"]
)
def hotel_cities():

    if hotels_df.empty:
        return jsonify([])

    cities = sorted(
        hotels_df["City"]
        .dropna()
        .astype(str)
        .str.title()
        .unique()
    )

    return jsonify(cities)


# ===================================================================
# HOTEL RECOMMENDATIONS
# ===================================================================

@app.route(
    "/hotels",
    methods=["GET"]
)
def get_hotels():

    city = request.args.get(
        "city",
        ""
    ).strip()

    tier = request.args.get(
        "tier",
        ""
    ).strip()

    min_rating = request.args.get(
        "min_rating",
        ""
    ).strip()

    if not city:

        return jsonify({
            "status": "error",
            "message":
                "Please select a location."
        }), 400

    matches = city_match(city).copy()

    if matches.empty:

        return jsonify({
            "city": city.title(),
            "count": 0,
            "hotels": []
        })

    # CHANGE #7
    # Remove duplicate hotel records BEFORE filtering.
    matches = remove_duplicate_hotels(
        matches
    )

    # CHANGE #3
    if tier in BUDGET_TIERS:

        low, high = BUDGET_TIERS[tier]

        matches = matches[
            (
                matches["Hotel_Price"]
                >= low
            )
            &
            (
                matches["Hotel_Price"]
                <= high
            )
        ]

    if min_rating:

        try:

            rating = float(
                min_rating
            )

            matches = matches[
                matches["Hotel_Rating"]
                >= rating
            ]

        except ValueError:
            pass

    matches = matches.sort_values(
        [
            "Hotel_Rating",
            "Hotel_Price"
        ],
        ascending=[
            False,
            True
        ]
    ).head(20)

    feature_cols = [
        c
        for c in hotels_df.columns
        if c.startswith("Feature_")
    ]

    results = []

    for _, row in matches.iterrows():

        features = [
            str(row[c]).strip()
            for c in feature_cols
            if (
                pd.notna(row.get(c))
                and
                str(row[c]).strip()
            )
        ]

        results.append({
            "hotel_name":
                str(row["Hotel_Name"]),
            "hotel_link":
                get_hotel_link(
                str(row["Hotel_Name"]),
            str(row["City"])
                ),
            "book_link":
                get_book_link(
                    str(row["Hotel_Name"]),
                    str(row["City"]),
                    row.get("Hotel_Link")
                ),
            "rating":
                (
                    round(
                        float(
                            row["Hotel_Rating"]
                        ),
                        1
                    )
                    if pd.notna(
                        row["Hotel_Rating"]
                    )
                    else None
                ),
            "city":
                str(
                    row["City"]
                ).title(),
            "price":
                round(
                    float(
                        row["Hotel_Price"]
                    ),
                    2
                ),
            "tier":
                get_tier(
                    row["Hotel_Price"]
                ),
            "features":
                features
        })

    return jsonify({
        "city": city.title(),
        "count": len(results),
        "hotels": results
    })
def get_hotel_link(hotel_name, city):
    query = quote_plus(f"{hotel_name} {city}")
    return f"https://www.google.com/search?q={query}"


def get_book_link(hotel_name, city, dataset_link):
    """
    Return the booking link straight from the dataset,
    with no substitution or fallback of any kind.
    """

    link = str(dataset_link).strip() if pd.notna(dataset_link) else ""

    if not link:
        return None

    return link

# ===================================================================
# BUDGET CALCULATOR
# ===================================================================

@app.route(
    "/calculate-budget",
    methods=["POST"]
)
def calculate_budget():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    city = str(
        data.get(
            "city",
            ""
        )
    ).strip()

    from_place = str(
        data.get(
            "from",
            ""
        )
    ).strip()

    mode = str(
        data.get(
            "mode",
            ""
        )
    ).strip()

    transport_class = str(
        data.get("transport_class")
        or
        (
            "Economy"
            if mode.lower() == "flight"
            else "SL"
        )
    ).strip()

    requested_tier = str(
        data.get(
            "tier",
            ""
        )
    ).strip()

    try:

        travelers = max(
            1,
            int(
                data.get(
                    "travelers",
                    1
                )
            )
        )

    except Exception:
        travelers = 1

    # CHANGE #6
    if (
        from_place
        and
        city
        and
        same_place(
            from_place,
            city
        )
    ):

        return jsonify({
            "status": "error",
            "message":
                "Source and destination cannot be the same."
        }), 400

    matches = (
        city_match(city)
        if city
        else hotels_df
    )

    # Remove duplicates before calculating hotel median.
    matches = remove_duplicate_hotels(
        matches
    )

    if matches.empty:

        hotel = 0
        hotel_count = 0

        tier = (
            requested_tier
            if requested_tier in BUDGET_TIERS
            else "Standard"
        )

    else:

        if requested_tier in BUDGET_TIERS:

            low, high = BUDGET_TIERS[
                requested_tier
            ]

            tier_matches = matches[
                (
                    matches["Hotel_Price"]
                    >= low
                )
                &
                (
                    matches["Hotel_Price"]
                    <= high
                )
            ]

            if not tier_matches.empty:
                matches = tier_matches

        hotel = round(
            float(
                matches["Hotel_Price"]
                .median()
            ),
            2
        )

        hotel_count = int(
            len(matches)
        )

        tier = (
            requested_tier
            if requested_tier in BUDGET_TIERS
            else
            get_tier(hotel)
        )

    # Existing non-transport estimates.
    food = 3000
    activities = 1500
    emergency = 2000

    transport_per_person = None
    transport_total = 0
    transport_source = "Not selected"

    if (
        from_place
        and
        city
        and
        mode
    ):

        (
            transport_per_person,
            transport_total,
            transport_source
        ) = calculate_transport_fare(
            from_place,
            city,
            mode,
            transport_class,
            travelers
        )

        if transport_total is None:
            transport_total = 0

    total = (
        hotel
        +
        food
        +
        transport_total
        +
        activities
        +
        emergency
    )

    return jsonify({
        "city": city,
        "from": from_place,
        "budget_level": tier,
        "travelers": travelers,
        "transport_mode": mode,
        "transport_class":
            transport_class,
        "hotel": hotel,
        "hotel_count":
            hotel_count,
        "food": food,
        "transport":
            round(
                transport_total,
                2
            ),
        "transport_per_person":
            transport_per_person,
        "transport_source":
            transport_source,
        "activities":
            activities,
        "emergency":
            emergency,
        "total":
            round(
                total,
                2
            )
    })


# ===================================================================
# AI PLANNER
# ===================================================================

@app.route(
    "/ai-itinerary",
    methods=["POST"]
)
def ai_itinerary():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    msg = (
        data.get("message")
        or ""
    ).strip()

    if not msg:
        return jsonify({
            "reply":
                "Please provide a destination and trip details."
        }), 400

    # ---------------------------------------------------------------
    # TRIP DURATION
    # ---------------------------------------------------------------

    days_match = re.search(
        r"\b(\d+)\s*(?:day|days)\b",
        msg.lower()
    )

    if days_match:
        days = int(
            days_match.group(1)
        )
    else:
        days = 3

    days = max(
        1,
        min(days, 30)
    )

    # ---------------------------------------------------------------
    # BUDGET
    # ---------------------------------------------------------------

    budget_input = data.get("budget")

    if budget_input not in (
        None,
        "",
        "null"
    ):

        try:
            budget = float(
                budget_input
            )

            if budget > 0:
                budget_given = True
            else:
                budget = None
                budget_given = False

        except (
            TypeError,
            ValueError
        ):

            budget = None
            budget_given = False

    else:

        budget = None
        budget_given = False

    # ---------------------------------------------------------------
    # TRAVELERS
    # ---------------------------------------------------------------

    try:

        travelers = max(
            1,
            int(
                data.get(
                    "travelers",
                    1
                )
            )
        )

    except (
        TypeError,
        ValueError
    ):

        travelers = 1

    # ---------------------------------------------------------------
    # SOURCE / DESTINATION
    # ---------------------------------------------------------------

    source = str(
        data.get("from")
        or ""
    ).strip()

    city = str(
        data.get("city")
        or ""
    ).strip()

    # ---------------------------------------------------------------
    # TRY TO DETECT DESTINATION FROM MESSAGE
    # ---------------------------------------------------------------

    if not city and not hotels_df.empty:

        for available_city in (
            hotels_df["City"]
            .dropna()
            .astype(str)
            .unique()
        ):

            if (
                normalize_city(
                    available_city
                )
                in
                normalize_city(msg)
            ):

                city = str(
                    available_city
                ).strip()

                break

    # ---------------------------------------------------------------
    # TRANSPORT INFORMATION
    # ---------------------------------------------------------------

    transport_mode = str(
        data.get("mode")
        or ""
    ).strip()

    transport_class = str(
        data.get("transport_class")
        or
        (
            "Economy"
            if transport_mode.lower() == "flight"
            else "SL"
        )
    ).strip()

    transport_per_person = 0
    transport_total = 0
    transport_source = "Not provided"

    # ---------------------------------------------------------------
    # CALCULATE TRANSPORT FROM SOURCE
    # ---------------------------------------------------------------

    if (
        source
        and
        city
        and
        transport_mode
    ):

        (
            calculated_per_person,
            calculated_total,
            calculated_source
        ) = calculate_transport_fare(
            source,
            city,
            transport_mode,
            transport_class,
            travelers
        )

        if (
            calculated_per_person is not None
            and
            calculated_per_person > 0
        ):

            transport_per_person = round(
                float(
                    calculated_per_person
                ),
                2
            )

            transport_total = round(
                float(
                    calculated_total
                ),
                2
            )

            transport_source = (
                calculated_source
            )

    # ---------------------------------------------------------------
    # HOTEL DATASET CONTEXT
    # ---------------------------------------------------------------

    hotel_context = ""

    hotel_price_per_night = 0
    hotel_name = ""
    hotel_rating = None

    if city and not hotels_df.empty:

        matches = city_match(city)

        matches = remove_duplicate_hotels(
            matches
        )

        if not matches.empty:

            # Cheapest suitable hotel first.
            matches = matches.sort_values(
                [
                    "Hotel_Price",
                    "Hotel_Rating"
                ],
                ascending=[
                    True,
                    False
                ]
            )

            cheapest = matches.iloc[0]

            hotel_name = str(
                cheapest["Hotel_Name"]
            )

            hotel_price_per_night = round(
                float(
                    cheapest["Hotel_Price"]
                ),
                2
            )

            if pd.notna(
                cheapest.get(
                    "Hotel_Rating"
                )
            ):

                hotel_rating = round(
                    float(
                        cheapest["Hotel_Rating"]
                    ),
                    1
                )

            recommended_hotels = []

            for _, row in matches.head(8).iterrows():

                rating_text = ""

                if pd.notna(
                    row.get(
                        "Hotel_Rating"
                    )
                ):

                    rating_text = (
                        f", rating "
                        f"{float(row['Hotel_Rating']):.1f}/5"
                    )

                recommended_hotels.append(
                    f"{row['Hotel_Name']} "
                    f"(₹{float(row['Hotel_Price']):.0f} "
                    f"per night"
                    f"{rating_text})"
                )

            hotel_context = (
                "\nHOTEL DATASET INFORMATION:\n"
                f"Destination: {city}\n"
                f"Hotels available: {len(matches)}\n"
                f"Cheapest suitable hotel: {hotel_name}\n"
                f"Cheapest hotel price: "
                f"₹{hotel_price_per_night} per night\n"
                f"Available hotel examples: "
                f"{'; '.join(recommended_hotels)}\n"
            )

    # ---------------------------------------------------------------
    # HOTEL COST CALCULATION
    # ---------------------------------------------------------------

    nights = max(
        1,
        days - 1
    )

    accommodation_total = round(
        hotel_price_per_night * nights,
        2
    )

    if travelers > 0:

        accommodation_per_person = round(
            accommodation_total / travelers,
            2
        )

    else:

        accommodation_per_person = (
            accommodation_total
        )

    # ---------------------------------------------------------------
    # BASE PER-PERSON EXPENSES
    # ---------------------------------------------------------------

    food_per_person = (
        1000 * days
    )

    local_transport_per_person = (
        500 * days
    )

    activities_per_person = (
        500 * days
    )

    miscellaneous_per_person = (
        300 * days
    )

    # ---------------------------------------------------------------
    # INITIAL BUDGET CALCULATION
    # ---------------------------------------------------------------

    minimum_per_person = round(
        accommodation_per_person
        +
        transport_per_person
        +
        food_per_person
        +
        local_transport_per_person
        +
        activities_per_person
        +
        miscellaneous_per_person,
        2
    )

    minimum_group_total = round(
        minimum_per_person * travelers,
        2
    )

    # ---------------------------------------------------------------
    # BUDGET STATUS
    # ---------------------------------------------------------------

    budget_per_person = None
    remaining_per_person = None
    budget_status = ""

    if budget_given:

        budget_per_person = round(
            budget / travelers,
            2
        )

        remaining_per_person = round(
            budget_per_person
            -
            minimum_per_person,
            2
        )

        if remaining_per_person >= 0:

            budget_status = (
                f"Within budget by "
                f"₹{remaining_per_person:.2f} per person."
            )

        else:

            budget_status = (
                f"Over budget by "
                f"₹{abs(remaining_per_person):.2f} "
                f"per person."
            )

    # ---------------------------------------------------------------
    # TRANSPORT CONTEXT FOR AI
    # ---------------------------------------------------------------

    transport_context = ""

    if (
        source
        and
        city
        and
        transport_mode
    ):

        if transport_per_person > 0:

            transport_context = (
                "\nTRANSPORT COST INFORMATION:\n"
                f"Source: {source}\n"
                f"Destination: {city}\n"
                f"Transport mode: {transport_mode}\n"
                f"Transport class: {transport_class}\n"
                f"Travelers: {travelers}\n"
                f"Verified backend fare per person: "
                f"₹{transport_per_person:.2f}\n"
                f"Verified backend fare for all travelers: "
                f"₹{transport_total:.2f}\n"
                f"Fare source: {transport_source}\n"
                "IMPORTANT: Use this transport fare in "
                "the final budget. Do not replace it "
                "with an invented fare.\n"
            )

        else:

            transport_context = (
                "\nTRANSPORT INFORMATION:\n"
                f"Source: {source}\n"
                f"Destination: {city}\n"
                f"Requested mode: {transport_mode}\n"
                "A reliable transport fare was not "
                "available from the backend.\n"
                "Do not invent an exact transport price.\n"
            )

    # ---------------------------------------------------------------
    # BUDGET CONTEXT FOR AI
    # ---------------------------------------------------------------

    budget_context = ""

    if budget_given:

        budget_context = (
            "\nUSER BUDGET:\n"
            f"Total group budget: ₹{budget:.2f}\n"
            f"Number of travelers: {travelers}\n"
            f"Budget per person: "
            f"₹{budget_per_person:.2f}\n"
            "The final plan should stay within "
            "this budget whenever reasonably possible.\n"
            "Do not spend the entire budget unnecessarily.\n"
        )

    else:

        budget_context = (
            "\nUSER BUDGET:\n"
            "No budget was provided.\n"
            "Determine a practical minimum budget "
            "using the supplied hotel and transport data "
            "and reasonable travel estimates.\n"
            "Do not assume a fixed ₹5,000 budget.\n"
        )

    # ---------------------------------------------------------------
    # AI PROMPT
    # ---------------------------------------------------------------

    prompt = (
        "You are SkyGuide India, a professional "
        "AI travel planning assistant.\n\n"

        "Your job is to create realistic, useful and "
        "financially consistent travel plans for India.\n\n"

        "TRIP REQUIREMENT:\n"
        f"Trip duration: {days} days\n"
        f"Number of travelers: {travelers}\n"
        f"Destination: {city or 'Determine from the user request'}\n"
        f"Source location: {source or 'Not provided'}\n\n"

        "IMPORTANT DAY REQUIREMENT:\n"
        f"You MUST provide exactly {days} days.\n"
        f"You MUST include Day 1 through Day {days}.\n"
        "Never stop before the requested number of days.\n"
        "Never reduce the requested duration.\n\n"

        f"USER REQUEST:\n{msg}\n\n"

        f"{hotel_context}\n"

        f"{transport_context}\n"

        f"{budget_context}\n"

        "BUDGET CALCULATION RULES:\n"
        "1. All final budget figures must be calculated "
        "on a per-person basis.\n"
        "2. Also show the total cost for all travelers.\n"
        "3. Do not confuse group totals with per-person totals.\n"
        "4. Accommodation must be divided among the travelers "
        "when showing the per-person cost.\n"
        "5. Include travel from the source location to the "
        "destination when a source and transport mode are provided.\n"
        "6. Use the backend transport fare exactly when one "
        "is supplied above.\n"
        "7. Do not invent a different transport fare.\n"
        "8. If no transport fare is available, clearly state "
        "that the fare could not be determined.\n"
        "9. Include food, local transport, activities and "
        "miscellaneous expenses.\n"
        "10. Use the actual hotel price supplied by the dataset.\n"
        "11. Do not invent hotel names when dataset hotels "
        "are available.\n"
        "12. Calculate accommodation using the number of nights.\n"
        "13. Calculate the final total mathematically.\n"
        "14. The sum of all budget categories must equal "
        "the stated total.\n"
        "15. If a user budget is provided, show the budget "
        "per person and whether the estimated trip is within "
        "or above that budget.\n"
        "16. If no budget is provided, provide a practical "
        "minimum estimated budget per person.\n\n"

        "ITINERARY RULES:\n"
        "1. Spread attractions logically across all requested days.\n"
        "2. Group nearby attractions together when practical.\n"
        "3. Avoid unnecessary repetition.\n"
        "4. Include important attractions and useful local experiences.\n"
        "5. Keep each day realistic rather than overcrowded.\n"
        "6. Mention specific times only when genuinely important, "
        "such as sunrise, sunset or evening markets.\n"
        "7. Do not assign a time to every activity.\n\n"

        "HOTEL RULES:\n"
        "1. Use the supplied hotel dataset.\n"
        "2. Do not invent hotel names if suitable dataset hotels exist.\n"
        "3. Prefer affordable suitable accommodation.\n"
        "4. Use the actual dataset price.\n\n"

        "WRITING RULES:\n"
        "1. Write in a professional travel-planning style.\n"
        "2. Do not use emojis.\n"
        "3. Do not use asterisks.\n"
        "4. Do not use hashtags.\n"
        "5. Do not use decorative Unicode characters.\n"
        "6. Do not use decorative separator lines.\n"
        "7. Do not use markdown tables.\n"
        "8. Do not use excessive punctuation.\n"
        "9. Use simple professional headings.\n"
        "10. Keep descriptions concise and useful.\n"
        "11. Use plain text formatting suitable for a web application.\n\n"

        "REQUIRED RESPONSE FORMAT:\n\n"

        "SKYGUIDE INDIA\n"
        f"{days}-Day Trip to [Destination]\n\n"

        "TRIP OVERVIEW\n"
        f"Travelers: {travelers}\n"
    )

    if budget_given:

        prompt += (
            f"Total Budget: ₹{budget:.2f}\n"
            f"Budget Per Person: "
            f"₹{budget_per_person:.2f}\n\n"
        )

    else:

        prompt += (
            "Budget: Not specified\n"
            f"Minimum Practical Budget Per Person: "
            f"₹{minimum_per_person:.2f}\n\n"
        )

    prompt += (
        "ACCOMMODATION\n"
        f"Hotel: {hotel_name or 'Suitable hotel from available dataset'}\n"
        f"Price Per Night: "
        f"₹{hotel_price_per_night:.2f}\n"
        f"Number of Nights: {nights}\n"
        f"Accommodation Cost Per Person: "
        f"₹{accommodation_per_person:.2f}\n\n"
    )

    if (
        source
        and
        city
        and
        transport_mode
    ):

        prompt += (
            "TRAVEL COST\n"
            f"From: {source}\n"
            f"To: {city}\n"
            f"Mode: {transport_mode}\n"
            f"Class: {transport_class}\n"
        )

        if transport_per_person > 0:

            prompt += (
                f"Travel Cost Per Person: "
                f"₹{transport_per_person:.2f}\n"
                f"Travel Cost For All Travelers: "
                f"₹{transport_total:.2f}\n\n"
            )

        else:

            prompt += (
                "Travel Cost Per Person: "
                "Not available\n\n"
            )

    # ---------------------------------------------------------------
    # DAILY ITINERARY FORMAT
    # ---------------------------------------------------------------

    for day_number in range(
        1,
        days + 1
    ):

        prompt += (
            f"DAY {day_number}\n"
            "Places to visit: [places]\n"
            "Activities: [activities]\n"
            "Best time: [only when genuinely important]\n"
            "Estimated Cost Per Person: ₹[amount]\n\n"
        )

    # ---------------------------------------------------------------
    # FINAL BUDGET FORMAT
    # ---------------------------------------------------------------

    prompt += (
    "BUDGET SUMMARY\n"
    f"Accommodation: ₹{accommodation_per_person:.2f}\n"
    f"Travel to Destination: ₹{transport_per_person:.2f}\n"
    f"Food: ₹{food_per_person:.2f}\n"
    f"Local Transport: ₹{local_transport_per_person:.2f}\n"
    f"Activities: ₹{activities_per_person:.2f}\n"
    f"Miscellaneous: ₹{miscellaneous_per_person:.2f}\n"
    f"Minimum Estimated Total: ₹{minimum_per_person:.2f}\n"
    )

    if budget_given:

        prompt += (
            f"Total Budget: ₹{budget:.2f}\n"
        )

    prompt += (
    f"Estimated Total For {travelers} Travelers: "
    f"₹{minimum_group_total:.2f}\n"
    )

    if budget_given:

        prompt += (
        f"Remaining Budget: ₹{remaining_per_person:.2f}\n"
        f"Budget Status: {budget_status}\n\n"

        "REMAINING BUDGET SUGGESTIONS\n"
        f"The user has ₹{remaining_per_person:.2f} remaining.\n"
        "Explain practical ways the user can use this remaining budget.\n"
        "Consider hotel upgrades, better food and dining, more "
        "comfortable transport, additional activities and experiences, "
        "shopping, and leisure.\n"
        "Give realistic suggestions based on the remaining amount.\n"
        "Do not force the user to spend the entire remaining budget.\n"
    )

    prompt += (
    "\nFINAL INSTRUCTION:\n"
    "Before producing the answer, verify every budget calculation.\n"
    "Do not add 'Per Person' to every budget category.\n"
    "Use simple category names such as Accommodation, Food, "
    "Local Transport, Activities and Miscellaneous.\n"
    "Show Total Budget and Remaining Budget clearly.\n"
    "If the user has remaining budget, include a section explaining "
    "what they can improve or add with that money, such as a better "
    "hotel, better food, more comfortable transport, additional "
    "activities, shopping or leisure.\n"
    "Do not invent or change the verified transport fare supplied "
    "by the backend.\n"
    "Do not force the user to spend the entire remaining budget.\n"
    "Do not output internal calculations or explanations.\n"
    "Return only the polished travel plan."
    )

    # ---------------------------------------------------------------
    # GROQ
    # ---------------------------------------------------------------

    try:

        chat = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="openai/gpt-oss-120b",
            temperature=0.2,
            max_tokens=5000
        )

        reply = (
            chat.choices[0]
            .message
            .content
            .strip()
        )

        # -----------------------------------------------------------
        # CLEAN UNWANTED MARKDOWN / SYMBOLS
        # -----------------------------------------------------------

        reply = re.sub(
            r"\*+",
            "",
            reply
        )

        reply = re.sub(
            r"^#+\s*",
            "",
            reply,
            flags=re.MULTILINE
        )

        reply = re.sub(
            r"^[•●▪◦]\s*",
            "",
            reply,
            flags=re.MULTILINE
        )

        reply = re.sub(
            r"^[─━═_*#]+$",
            "",
            reply,
            flags=re.MULTILINE
        )

        reply = re.sub(
            r"\n{3,}",
            "\n\n",
            reply
        )

        return jsonify({
            "reply": reply.strip()
        })

    except Exception as e:

        print(
            f"[error] Groq call failed: {e}"
        )

        return jsonify({
            "reply":
                "The travel planner is temporarily unavailable. "
                "Please try again."
        }), 502


# ===================================================================
# CHAT HISTORY
# ===================================================================

def load_chat_history():

    try:

        with open(
            CHAT_HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:
        return {}


def save_chat_history(data):

    temp = (
        CHAT_HISTORY_FILE
        + ".tmp"
    )

    with open(
        temp,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    os.replace(
        temp,
        CHAT_HISTORY_FILE
    )


@app.route(
    "/chat-history",
    methods=["GET"]
)
def get_chat_history():

    email = session.get("user_email")

    print(
        f"[history GET] Session email: {email}"
    )

    if not email:

        return jsonify({
            "status": "error",
            "message": "Login required.",
            "chats": []
        }), 401

    all_history = load_chat_history()

    chats = all_history.get(
        email,
        []
    )

    # Older saved chats have no id. Assign one and persist it
    # so delete works on chats created before this change.
    changed = False

    for chat in chats:

        if not chat.get("id"):
            chat["id"] = secrets.token_hex(8)
            changed = True

    if changed:
        all_history[email] = chats
        save_chat_history(all_history)

    return jsonify({
        "status": "success",
        "chats": chats[-30:][::-1]
    })


@app.route(
    "/chat-history",
    methods=["POST"]
)
def add_chat_history():

    email = session.get("user_email")

    print(
        f"[history POST] Session email: {email}"
    )

    if not email:

        return jsonify({
            "status": "error",
            "message": "Login required."
        }), 401

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    message = (
        data.get("message")
        or ""
    ).strip()

    reply = (
        data.get("reply")
        or ""
    ).strip()

    if not message or not reply:

        return jsonify({
            "status": "error",
            "message":
                "Message and reply are required."
        }), 400

    all_history = load_chat_history()

    user_history = all_history.setdefault(
        email,
        []
    )

    user_history.append({
        "id": secrets.token_hex(8),
        "title": message[:60],
        "message": message,
        "reply": reply,
        "created_at":
            datetime.now().strftime(
                "%d %b %Y, %I:%M %p"
            )
    })

    all_history[email] = user_history[-30:]

    save_chat_history(all_history)

    print(
        f"[history POST] Saved chat for {email}"
    )

    print(
        f"[history POST] Total chats: "
        f"{len(all_history[email])}"
    )

    return jsonify({
        "status": "success",
        "message": "Chat history saved.",
        "count": len(all_history[email])
    })


@app.route(
    "/chat-history/<chat_id>",
    methods=["DELETE"]
)
def delete_chat(chat_id):

    email = session.get("user_email")

    if not email:

        return jsonify({
            "status": "error",
            "message": "Login required."
        }), 401

    all_history = load_chat_history()

    user_history = all_history.get(
        email,
        []
    )

    remaining = [
        c
        for c in user_history
        if str(c.get("id")) != str(chat_id)
    ]

    if len(remaining) == len(user_history):

        return jsonify({
            "status": "error",
            "message": "Chat not found."
        }), 404

    all_history[email] = remaining

    save_chat_history(all_history)

    print(
        f"[history DELETE] Removed {chat_id} for {email}"
    )

    return jsonify({
        "status": "success",
        "deleted": chat_id,
        "count": len(remaining)
    })


@app.route(
    "/chat-history",
    methods=["DELETE"]
)
def clear_chat_history():

    email = session.get("user_email")

    if not email:

        return jsonify({
            "status": "error",
            "message": "Login required."
        }), 401

    all_history = load_chat_history()

    all_history[email] = []

    save_chat_history(all_history)

    print(
        f"[history DELETE] Cleared all chats for {email}"
    )

    return jsonify({
        "status": "success",
        "cleared": True
    })


# ===================================================================
# HEALTH
# ===================================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "ok",

        "hotels_loaded":
            int(
                len(hotels_df)
            ),

        "cities_loaded":
            int(
                hotels_df["City"]
                .nunique()
                if not hotels_df.empty
                else 0
            ),

        "flight_records":
            int(
                len(flights_df)
            ),

        "train_schedule_records":
            int(
                len(train_schedule_df)
            ),

        "train_fare_records":
            int(
                len(train_fares_df)
            ),

        "dataset_priority":
            "LOCAL_FIRST_S3_FALLBACK",

        "hotel_budget_tiers":
            list(
                BUDGET_TIERS.keys()
            )
    })
@app.route("/auth/google", methods=["POST"])
def auth_google():
    data = request.get_json(silent=True) or {}
    credential = str(data.get("credential") or "").strip()

    if not credential:
        return jsonify({
            "status": "error",
            "message": "Google credential is missing."
        }), 400

    try:
        google_user = id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        email = str(google_user.get("email") or "").strip().lower()
        name = str(
            google_user.get("name")
            or google_user.get("given_name")
            or "Google User"
        ).strip()

        if not email:
            return jsonify({
                "status": "error",
                "message": "Google did not provide an email address."
            }), 400

        if not google_user.get("email_verified"):
            return jsonify({
                "status": "error",
                "message": "The Google email address is not verified."
            }), 401

        users = load_users()

        # Create the local SkyGuide account automatically if needed.
        if email not in users:
            users[email] = {
                "name": name,
                "password": generate_password_hash(
                    secrets.token_urlsafe(32)
                ),
                "google_login": True
            }
            save_users(users)
        else:
            # Keep the existing account but update the display name.
            users[email]["name"] = name or users[email].get("name", "Google User")
            users[email]["google_login"] = True
            save_users(users)

        session.clear()
        session["user_email"] = email
        session["user_name"] = name or users[email].get("name", "Google User")

        return jsonify({
            "status": "success",
            "email": email,
            "name": session["user_name"]
        })

    except ValueError:
        return jsonify({
            "status": "error",
            "message": "Invalid or expired Google login credential."
        }), 401

    except Exception as e:
        print(f"[error] Google authentication failed: {e}")
        return jsonify({
            "status": "error",
            "message": "Google login could not be completed."
        }), 500


# ===================================================================
# START SERVER
# ===================================================================

if __name__ == "__main__":

    print("")
    print("=" * 60)
    print("SkyGuide India Backend")
    print("=" * 60)
    print("Dataset priority: LOCAL/OFFLINE → S3 fallback")
    print("Server: http://127.0.0.1:8000")
    print("=" * 60)
    print("")

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False
    )