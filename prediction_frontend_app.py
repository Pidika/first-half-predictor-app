
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import warnings
from datetime import datetime
from team_name_mapper import normalize_team_name
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="First-Half Goal Predictor",
    page_icon="⚽",
    layout="wide"
)

# --- Configuration ---
MODEL_FILE = 'final_prediction_model.joblib'
HISTORICAL_DATA_FILE = 'features_engineered_data.csv'
TRAINING_COLUMNS_FILE = 'final_data.csv'
# --- The Odds API Configuration ---
# This will be loaded from Streamlit's secrets manager
try:
    API_KEY = st.secrets["API_KEY"]
except (FileNotFoundError, KeyError):
    API_KEY = "YOUR API_KEY" 
# A list of major European leagues to fetch fixtures for.
LEAGUES = {
    'soccer_epl': 'English Premier League',
    'soccer_spain_la_liga': 'Spanish La Liga',
    'soccer_germany_bundesliga': 'German Bundesliga',
    'soccer_italy_serie_a': 'Italian Serie A',
    'soccer_france_ligue_one': 'French Ligue 1',
    'soccer_portugal_primeira_liga': 'Portuguese Primeira Liga',
    'soccer_spl': 'Scottish Premiership',
    'soccer_netherlands_eredivisie': 'Dutch Eredivisie'
}
# The free tier only reliably provides the 'h2h' (head-to-head/match winner) market.
PRIMARY_MARKET_FOR_API_CALL = 'h2h' 


# --- Asset Loading with Caching ---
@st.cache_resource
def load_assets(model_path, historical_data_path, training_cols_path):
    """
    Loads the trained model, historical data, and the list of feature names.
    """
    try:
        model = joblib.load(model_path)
        historical_df = pd.read_csv(historical_data_path, parse_dates=['Date'])
        historical_df = historical_df.sort_values(by='Date').reset_index(drop=True)
        training_cols_df = pd.read_csv(training_cols_path)
        model_features = training_cols_df.columns.tolist()
        return model, historical_df, model_features
    except FileNotFoundError as e:
        st.error(f"ERROR: Could not find a required file: {e}. Please ensure all required .joblib and .csv files are present.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading assets: {e}")
        return None, None, None

# --- Live Data Fetching ---
@st.cache_data(ttl=3600) # Cache the data for 1 hour
def fetch_live_fixtures(api_key):
    """
    Fetches live upcoming fixtures and h2h odds from The Odds API.
    """
    st.write("Fetching live fixtures...")
    fixtures_by_league = {}
    for league_key, league_name in LEAGUES.items():
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds/?apiKey={api_key}&regions=eu&markets={PRIMARY_MARKET_FOR_API_CALL}&oddsFormat=decimal"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            current_league_fixtures = []
            for match in data:
                home_team_api = match.get('home_team')
                away_team_api = match.get('away_team')
                home_team = normalize_team_name(home_team_api)
                away_team = normalize_team_name(away_team_api)
                commence_time = datetime.fromisoformat(match.get('commence_time').replace('Z', '+00:00'))

                h2h_odds = {'home': 0, 'away': 0, 'draw': 0}
                if match.get('bookmakers'):
                    for bookmaker in match['bookmakers']:
                        for market in bookmaker.get('markets', []):
                            if market.get('key') == 'h2h':
                                for outcome in market.get('outcomes', []):
                                    if outcome['name'] == home_team_api and outcome['price'] > h2h_odds['home']:
                                        h2h_odds['home'] = outcome['price']
                                    elif outcome['name'] == away_team_api and outcome['price'] > h2h_odds['away']:
                                        h2h_odds['away'] = outcome['price']
                                    elif outcome['name'] == 'Draw' and outcome['price'] > h2h_odds['draw']:
                                        h2h_odds['draw'] = outcome['price']

                if all(val > 0 for val in h2h_odds.values()):
                    current_league_fixtures.append({
                        "id": match['id'],
                        "display_name": f"{home_team} vs {away_team} ({commence_time.strftime('%b %d, %H:%M')})",
                        "home_team": home_team,
                        "away_team": away_team,
                        "h2h_home": h2h_odds['home'],
                        "h2h_draw": h2h_odds['draw'],
                        "h2h_away": h2h_odds['away'],
                    })
            
            if current_league_fixtures:
                fixtures_by_league[league_name] = pd.DataFrame(current_league_fixtures)

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch data for {league_name}: {e}")
            continue
            
    if not fixtures_by_league:
        return None
        
    return fixtures_by_league


# --- Core Logic Functions ---
def get_latest_team_features(team_name, historical_df):
    last_match = historical_df[(historical_df['HomeTeam'] == team_name) | (historical_df['AwayTeam'] == team_name)].tail(1)
    if last_match.empty: return None
    prefix = 'home_' if last_match.iloc[0]['HomeTeam'] == team_name else 'away_'
    features = {}
    feature_suffixes = ['elo', 'attack_strength', 'defense_strength', 'ewma_goals_scored', 'ewma_goals_conceded', 'ewma_shots', 'ewma_shots_on_target', 'ewma_corners']
    for suffix in feature_suffixes:
        col_name = f'{prefix}{suffix}'
        features[suffix] = last_match.iloc[0].get(col_name, np.nan)
    return features

def create_prediction_dataframe(home_features, away_features, h2h_odds, model_feature_list):
    feature_dict = {}
    for key, value in home_features.items(): feature_dict[f'home_{key}'] = value
    for key, value in away_features.items(): feature_dict[f'away_{key}'] = value
    odds_features = {'AvgH': h2h_odds['h2h_home'], 'AvgD': h2h_odds['h2h_draw'], 'AvgA': h2h_odds['h2h_away']}
    feature_dict.update(odds_features)
    
    prediction_df = pd.DataFrame([feature_dict])
    for col in model_feature_list:
        if col not in prediction_df.columns:
            prediction_df[col] = np.nan
    return prediction_df[model_feature_list]

def get_implied_probability(decimal_odds):
    return 1 / decimal_odds if decimal_odds > 1 else 0

def calculate_expected_value(model_prob, decimal_odds):
    if decimal_odds <= 1: return -1
    profit_per_dollar = decimal_odds - 1
    return (model_prob * profit_per_dollar) - ((1 - model_prob) * 1)

def process_multiple_matches(matches_to_predict, model, historical_df, model_features):
    results = []
    for match_data in matches_to_predict:
        home_team = match_data['home_team']
        away_team = match_data['away_team']
        over_05_odds = match_data['over_05_odds']
        h2h_odds = match_data['h2h_odds']

        home_features = get_latest_team_features(home_team, historical_df)
        away_features = get_latest_team_features(away_team, historical_df)
        
        if home_features is None or away_features is None:
            st.warning(f"Could not find historical data for {home_team} or {away_team}. Skipping.")
            continue

        prediction_df = create_prediction_dataframe(home_features, away_features, h2h_odds, model_features)
        
        try:
            probability_of_goal = model.predict_proba(prediction_df)[0, 1]
            market_prob = get_implied_probability(over_05_odds)
            ev = calculate_expected_value(probability_of_goal, over_05_odds)
            
            results.append({
                "Match": f"{home_team} vs {away_team}",
                "Market Odds (Over 0.5)": over_05_odds,
                "Model Probability": probability_of_goal,
                "Market Probability": market_prob,
                "Expected Value (EV)": ev,
                "Recommendation": "Value Bet" if ev > 0 else "No Value"
            })
        except Exception as e:
            st.error(f"An error occurred while predicting {home_team} vs {away_team}: {e}")

    return pd.DataFrame(results)

# --- Streamlit App UI ---
st.title("⚽ First-Half Goal Value Bet Finder")
st.markdown("This tool uses a trained ML model to analyze upcoming football matches and identify potential **value bets** in the 'Over 0.5 First-Half Goals' market.")

model, historical_data, model_features = load_assets(MODEL_FILE, HISTORICAL_DATA_FILE, TRAINING_COLUMNS_FILE)

if model and historical_data is not None:
    if API_KEY == "YOUR_API_KEY_HERE":
        st.warning("Please add your The Odds API Key to the script to fetch live fixtures.")
        st.info("For deployment, it's best to use Streamlit's Secrets Management. Modify the code to `API_KEY = st.secrets['API_KEY']` and add the secret in your app's settings.")
    else:
        live_fixtures_by_league = fetch_live_fixtures(API_KEY)

        if live_fixtures_by_league is not None and not live_fixtures_by_league == {}:
            st.success(f"Successfully fetched {sum(len(df) for df in live_fixtures_by_league.values())} upcoming matches across {len(live_fixtures_by_league)} leagues.")
            st.header("Analyze Matches")
            st.markdown("Enter the 'Over 0.5 First-Half Goals' odds for the matches you wish to analyze, then click the button at the bottom.")

            matches_to_process = []
            
            with st.form("matches_form"):
                for league_name, fixtures_df in live_fixtures_by_league.items():
                    with st.expander(f"**{league_name}** ({len(fixtures_df)} matches)", expanded=False):
                        for index, row in fixtures_df.iterrows():
                            st.subheader(row['display_name'])
                            col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
                            
                            with col1:
                                st.metric(label=f"{row['home_team']} (Home)", value=f"{row['h2h_home']:.2f}")
                            with col2:
                                st.metric(label="Draw", value=f"{row['h2h_draw']:.2f}")
                            with col3:
                                st.metric(label=f"{row['away_team']} (Away)", value=f"{row['h2h_away']:.2f}")

                            with col4:
                                over_05_odds = st.number_input(
                                    "Over 0.5 FHG Odds", 
                                    min_value=1.0, 
                                    value=1.0, 
                                    step=0.01, 
                                    key=row['id'],
                                    help="Enter the decimal odds for Over 0.5 goals in the first half."
                                )
                            
                            if over_05_odds > 1.0:
                                matches_to_process.append({
                                    "home_team": row['home_team'],
                                    "away_team": row['away_team'],
                                    "over_05_odds": over_05_odds,
                                    "h2h_odds": {
                                        "h2h_home": row['h2h_home'],
                                        "h2h_draw": row['h2h_draw'],
                                        "h2h_away": row['h2h_away']
                                    }
                                })
                            st.divider()
                
                form_col1, form_col2 = st.columns(2)
                with form_col1:
                    submitted = st.form_submit_button(
                        "Analyze Matches with Entered Odds", 
                        type="primary", 
                        use_container_width=True
                    )
                with form_col2:
                    if st.form_submit_button("Clear All Selections", use_container_width=True):
                        st.rerun()

            if submitted:
                if not matches_to_process:
                    st.warning("Please enter valid odds (>1.0) for at least one match to analyze.")
                else:
                    with st.spinner("Running predictions..."):
                        results_df = process_multiple_matches(matches_to_process, model, historical_data, model_features)
                    
                    if not results_df.empty:
                        st.header("Prediction Results")
                        
                        def style_ev(val):
                            return 'background-color: #2E8B57; color: white;' if val > 0 else ''

                        styled_df = results_df.style \
                           .format({
                                "Market Odds (Over 0.5)": "{:.2f}",
                                "Model Probability": "{:.2%}",
                                "Market Probability": "{:.2%}",
                                "Expected Value (EV)": "{:+.2%}"
                            }) \
                           .applymap(style_ev, subset=['Expected Value (EV)'])

                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.error("Could not fetch live fixtures. Please check your API key and network connection. Note: There may be no upcoming matches available right now.")
else:
    st.warning("Application could not start because required data and model files were not loaded.")

st.markdown("---")
st.info("Disclaimer: This tool is for educational and informational purposes only. Predictions are not guaranteed. Please gamble responsibly and only bet what you can afford to lose.")
