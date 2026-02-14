"""Script to generate the FotMob-based Saudi League prediction notebook."""
import json
from pathlib import Path

cells = []
SEASON_LABEL = "2025/26"


def _normalize_text(source):
    # Keep season labels consistent across all markdown/code cells.
    return source.replace("2024/25", SEASON_LABEL).replace("2025/26", SEASON_LABEL)

def md(source):
    source = _normalize_text(source)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n")  # will rejoin later
    })

def code(source):
    source = _normalize_text(source)
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": source.split("\n"),
        "outputs": [],
        "execution_count": None
    })

# ============================================================
# SECTION 1: SETUP & IMPORTS
# ============================================================
md("# Saudi Pro League Winner Prediction 2025/26\n### Powered by FotMob Data & Machine Learning")

md("---\n## 1. Setup & Imports")

code("""import requests
import pandas as pd
import numpy as np
import time
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
import xgboost as xgb

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Dark theme for YouTube-ready visuals
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 120,
    'axes.facecolor': '#1a1a2e',
    'figure.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.color': '#e94560'
})

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Referer': 'https://www.fotmob.com/'
}

LEAGUE_ID = 536  # Saudi Pro League
print("Setup complete!")""")

# ============================================================
# SECTION 2: DATA COLLECTION
# ============================================================
md("---\n## 2. Data Collection from FotMob API")

md("### 2a. Fetch League Overview (Standings, Stat Links, Fixtures)")

code("""# Fetch main league data
league_url = f"https://www.fotmob.com/api/leagues?id={LEAGUE_ID}"
print(f"Fetching league overview from: {league_url}")
resp = requests.get(league_url, headers=HEADERS)
print(f"Status: {resp.status_code}")
league_data = resp.json()

# Extract standings
standings_raw = league_data['table'][0]['data']['table']['all']
standings_df = pd.DataFrame(standings_raw)
print(f"\\nFound {len(standings_df)} teams in standings")
print(standings_df[['name', 'played', 'wins', 'draws', 'losses', 'pts']].to_string(index=False))""")

md("### 2b. Scrape ALL Team Stat Categories (27 categories)")

code("""# Get all available stat categories
stat_categories = league_data['stats']['teams']
print(f"Available stat categories: {len(stat_categories)}")
for i, cat in enumerate(stat_categories):
    print(f"  {i+1}. {cat['header']}")""")

code("""# Fetch every stat category
all_stats = {}
season_id = None

for i, cat in enumerate(stat_categories):
    cat_name = cat['header']
    # Get the fetch URL from statLink or fetchAllUrl
    stat_link = cat.get('fetchAllUrl', '')
    if not stat_link and 'statLink' in cat:
        stat_link = cat['statLink']

    if not stat_link:
        print(f"  Skipping {cat_name} - no URL found")
        continue

    # Build full URL
    if stat_link.startswith('/'):
        url = f"https://www.fotmob.com{stat_link}"
    elif not stat_link.startswith('http'):
        url = f"https://www.fotmob.com/{stat_link}"
    else:
        url = stat_link

    # Extract season ID from URL if we don't have it yet
    if season_id is None:
        parts = stat_link.split('/')
        for j, p in enumerate(parts):
            if p == 'season' and j + 1 < len(parts):
                season_id = parts[j + 1]
                break

    print(f"  [{i+1}/{len(stat_categories)}] Fetching {cat_name}...")
    try:
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            data = r.json()
            # Extract stat list
            if 'TopLists' in data:
                for top_list in data['TopLists']:
                    stat_key = top_list.get('Title', cat_name)
                    stat_list = top_list.get('StatList', [])
                    all_stats[stat_key] = stat_list
                    print(f"    -> {stat_key}: {len(stat_list)} teams")
            else:
                print(f"    -> Unexpected format")
        else:
            print(f"    -> Status {r.status_code}")
    except Exception as e:
        print(f"    -> Error: {e}")

    time.sleep(1.5)  # Be polite

print(f"\\nTotal stat categories fetched: {len(all_stats)}")
print(f"Season ID: {season_id}")""")

md("### 2c. Fetch Match Fixtures & Results")

code("""# Extract fixtures
fixtures_raw = league_data.get('matches', {}).get('allMatches', [])
if not fixtures_raw:
    fixtures_raw = league_data.get('fixtures', {}).get('allMatches', [])
if not fixtures_raw:
    # Try to find fixtures in the data
    for key in league_data:
        if 'match' in key.lower() or 'fixture' in key.lower():
            print(f"Found fixtures key: {key}")

# Parse fixtures
fixtures = []
for match in fixtures_raw:
    try:
        home = match.get('home', {})
        away = match.get('away', {})
        fixtures.append({
            'home_team': home.get('name', ''),
            'away_team': away.get('name', ''),
            'home_score': home.get('score'),
            'away_score': away.get('score'),
            'status': match.get('status', {}).get('finished', False),
            'round': match.get('round', '')
        })
    except:
        continue

fixtures_df = pd.DataFrame(fixtures)
if len(fixtures_df) > 0:
    played_df = fixtures_df[fixtures_df['status'] == True]
    upcoming_df = fixtures_df[fixtures_df['status'] == False]
    print(f"Total fixtures: {len(fixtures_df)}")
    print(f"Played: {len(played_df)}, Upcoming: {len(upcoming_df)}")
else:
    print("Fixtures not found in expected location - will derive from standings")
    played_df = pd.DataFrame()
    upcoming_df = pd.DataFrame()""")

md("### 2d. Fetch Historical Seasons Data")

code("""# Get historical season links
season_links = league_data.get('stats', {}).get('seasonStatLinks', [])
print(f"Historical seasons available: {len(season_links)}")
for s in season_links:
    print(f"  - {s.get('Name', 'Unknown')}: TournamentId={s.get('TournamentId', 'N/A')}")

# Fetch last 3-5 historical seasons standings
historical_standings = {}
current_season_name = season_links[0]['Name'] if season_links else "2024/2025"

for s in season_links[1:6]:  # Skip current, get up to 5 past seasons
    s_name = s.get('Name', '')
    t_id = s.get('TournamentId', '')
    if not t_id:
        continue

    url = f"https://www.fotmob.com/api/leagues?id={LEAGUE_ID}&season={t_id}"
    print(f"Fetching {s_name} (TournamentId={t_id})...")
    try:
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            hist_data = r.json()
            hist_table = hist_data.get('table', [{}])[0].get('data', {}).get('table', {}).get('all', [])
            if hist_table:
                historical_standings[s_name] = pd.DataFrame(hist_table)
                print(f"  -> {len(hist_table)} teams")
            else:
                print(f"  -> No table data found")
        else:
            print(f"  -> Status {r.status_code}")
    except Exception as e:
        print(f"  -> Error: {e}")
    time.sleep(2)

print(f"\\nHistorical seasons loaded: {len(historical_standings)}")""")

# ============================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================
md("---\n## 3. Data Cleaning & Feature Engineering")

md("### 3a. Build Master DataFrame from Standings")

code("""# Start with standings as base
master_df = standings_df[['name', 'played', 'wins', 'draws', 'losses', 'pts']].copy()
master_df.columns = ['team', 'played', 'wins', 'draws', 'losses', 'pts']

# Parse goals from scoresStr (format: "GF-GA")
if 'scoresStr' in standings_df.columns:
    master_df['goals_for'] = standings_df['scoresStr'].apply(lambda x: int(x.split('-')[0]) if isinstance(x, str) and '-' in x else 0)
    master_df['goals_against'] = standings_df['scoresStr'].apply(lambda x: int(x.split('-')[1]) if isinstance(x, str) and '-' in x else 0)
elif 'goalConDiff' in standings_df.columns:
    master_df['goal_diff'] = standings_df['goalConDiff'].astype(int)

# Goal difference
if 'goals_for' in master_df.columns:
    master_df['goal_diff'] = master_df['goals_for'] - master_df['goals_against']

# Basic per-match rates
master_df['pts_per_match'] = master_df['pts'] / master_df['played']
master_df['win_rate'] = master_df['wins'] / master_df['played']
master_df['draw_rate'] = master_df['draws'] / master_df['played']
master_df['loss_rate'] = master_df['losses'] / master_df['played']

if 'goals_for' in master_df.columns:
    master_df['goals_per_match'] = master_df['goals_for'] / master_df['played']
    master_df['goals_conceded_per_match'] = master_df['goals_against'] / master_df['played']

print(f"Master DataFrame: {master_df.shape}")
master_df.head()""")

md("### 3b. Merge All Scraped Stats into Master DataFrame")

code("""# Map scraped stats to master_df
stat_mapping = {}
for stat_key, stat_list in all_stats.items():
    # Create a clean column name
    col_name = stat_key.lower().replace(' ', '_').replace('/', '_per_').replace('-', '_')
    col_name = col_name.replace('(', '').replace(')', '').replace('%', 'pct')

    # Build mapping from team name to stat value
    team_stats = {}
    for entry in stat_list:
        team_name = entry.get('ParticipantName', entry.get('TeamName', ''))
        stat_val = entry.get('StatValue', entry.get('SubStatValue', ''))

        # Clean stat value - handle percentage strings and comma numbers
        if isinstance(stat_val, str):
            stat_val = stat_val.replace('%', '').replace(',', '').strip()
            try:
                stat_val = float(stat_val)
            except:
                continue

        if team_name:
            team_stats[team_name] = stat_val

    if team_stats:
        stat_mapping[col_name] = team_stats

print(f"Stat columns to add: {len(stat_mapping)}")

# Merge stats with fuzzy matching
for col_name, team_stats in stat_mapping.items():
    values = []
    for team in master_df['team']:
        # Try exact match first
        if team in team_stats:
            values.append(team_stats[team])
        else:
            # Try partial matching
            matched = False
            for stat_team, val in team_stats.items():
                if team.lower() in stat_team.lower() or stat_team.lower() in team.lower():
                    values.append(val)
                    matched = True
                    break
            if not matched:
                values.append(np.nan)

    master_df[col_name] = values

print(f"\\nMaster DataFrame shape: {master_df.shape}")
print(f"Columns: {list(master_df.columns)}")
print(f"\\nMissing values per column:")
print(master_df.isnull().sum()[master_df.isnull().sum() > 0])""")

md("### 3c. Engineer Advanced Features")

code("""# xG-based features (if available)
xg_cols = [c for c in master_df.columns if 'xg' in c.lower() or 'expected' in c.lower()]
print(f"xG-related columns found: {xg_cols}")

# Try to identify xG and xGA columns
xg_col = None
xga_col = None
for c in xg_cols:
    if 'against' in c.lower() or 'conceded' in c.lower() or 'xga' in c.lower():
        xga_col = c
    elif 'xg' in c.lower():
        if xg_col is None:
            xg_col = c

if xg_col and xga_col:
    master_df['xg_diff'] = master_df[xg_col] - master_df[xga_col]
    master_df['xg_overperformance'] = master_df.get('goals_for', master_df.get('goals_per_match', 0)) - master_df[xg_col]
    print(f"Created xG features using {xg_col} and {xga_col}")
elif xg_col:
    print(f"Only found xG column: {xg_col}, no xGA column")

# Per-match normalization for counting stats
counting_cols = [c for c in master_df.columns if master_df[c].dtype in ['float64', 'int64']
                 and c not in ['played', 'wins', 'draws', 'losses', 'pts', 'pts_per_match',
                              'win_rate', 'draw_rate', 'loss_rate', 'goals_per_match',
                              'goals_conceded_per_match', 'goal_diff']]

print(f"\\nFeature columns: {len(counting_cols)}")
print(f"Total features in master_df: {master_df.shape[1]}")
master_df.head(3)""")

md("### 3d. Add Historical Features (Low Weight)")

code("""# Calculate historical pedigree
historical_features = {}

for team in master_df['team']:
    positions = []
    titles = 0
    was_champion_last = 0

    for i, (season, hist_df) in enumerate(historical_standings.items()):
        for idx, row in hist_df.iterrows():
            hist_team = row.get('name', '')
            if team.lower() in hist_team.lower() or hist_team.lower() in team.lower():
                pos = idx + 1  # Position (1-indexed)
                positions.append(pos)
                if pos == 1:
                    titles += 1
                if i == 0 and pos == 1:  # Most recent historical season
                    was_champion_last = 1
                break

    avg_position = np.mean(positions) if positions else 10  # Default mid-table
    historical_features[team] = {
        'hist_avg_position': avg_position,
        'hist_titles': titles,
        'hist_was_champion_last': was_champion_last,
        'hist_seasons_found': len(positions)
    }

hist_df = pd.DataFrame(historical_features).T
hist_df.index.name = 'team'
hist_df = hist_df.reset_index()

master_df = master_df.merge(hist_df, on='team', how='left')

# Fill NaN historical features with defaults
master_df['hist_avg_position'] = master_df['hist_avg_position'].fillna(10)
master_df['hist_titles'] = master_df['hist_titles'].fillna(0)
master_df['hist_was_champion_last'] = master_df['hist_was_champion_last'].fillna(0)
master_df['hist_seasons_found'] = master_df['hist_seasons_found'].fillna(0)

print("Historical features added:")
print(master_df[['team', 'hist_avg_position', 'hist_titles', 'hist_was_champion_last']].to_string(index=False))""")

md("### 3e. Calculate Recent Form (from fixtures)")

code("""# Calculate form from last 5 matches
if len(played_df) > 0:
    form_data = {}
    for team in master_df['team']:
        team_matches = played_df[
            (played_df['home_team'].str.contains(team, case=False, na=False)) |
            (played_df['away_team'].str.contains(team, case=False, na=False))
        ].tail(5)

        form_points = 0
        form_goals = 0
        form_conceded = 0
        matches_found = 0

        for _, match in team_matches.iterrows():
            if match['home_score'] is not None and match['away_score'] is not None:
                try:
                    hs = int(match['home_score'])
                    as_ = int(match['away_score'])
                except:
                    continue

                matches_found += 1
                is_home = team.lower() in str(match['home_team']).lower()

                if is_home:
                    form_goals += hs
                    form_conceded += as_
                    if hs > as_: form_points += 3
                    elif hs == as_: form_points += 1
                else:
                    form_goals += as_
                    form_conceded += hs
                    if as_ > hs: form_points += 3
                    elif hs == as_: form_points += 1

        form_data[team] = {
            'form_points_last5': form_points,
            'form_goals_last5': form_goals,
            'form_conceded_last5': form_conceded,
            'form_matches': matches_found,
            'form_ppg': form_points / max(matches_found, 1)
        }

    form_df = pd.DataFrame(form_data).T
    form_df.index.name = 'team'
    form_df = form_df.reset_index()
    master_df = master_df.merge(form_df, on='team', how='left')
    print("Form features added from fixture data")
else:
    print("No fixture data available - skipping form features")
    master_df['form_ppg'] = master_df['pts_per_match']  # Use overall as proxy

print(f"\\nFinal Master DataFrame: {master_df.shape}")
print(f"\\nAll columns ({len(master_df.columns)}):")
for i, col in enumerate(master_df.columns):
    print(f"  {i+1}. {col}")""")

# ============================================================
# SECTION 4: EDA
# ============================================================
md("---\n## 4. Exploratory Data Analysis")

md("### 4a. Current Standings")

code("""# Standings bar chart
fig, ax = plt.subplots(figsize=(14, 10))
sorted_df = master_df.sort_values('pts', ascending=True)

colors = ['#e94560' if i >= len(sorted_df) - 3 else '#0f3460' for i in range(len(sorted_df))]
bars = ax.barh(sorted_df['team'], sorted_df['pts'], color=colors, edgecolor='white', linewidth=0.5)

# Add value labels
for bar, pts in zip(bars, sorted_df['pts']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{int(pts)}', va='center', fontweight='bold', color='white', fontsize=11)

ax.set_xlabel('Points', fontweight='bold')
ax.set_title('Saudi Pro League 2024/25 - Current Standings', fontweight='bold', fontsize=18, color='#e94560')
ax.legend(handles=[
    mpatches.Patch(color='#e94560', label='Top 3'),
    mpatches.Patch(color='#0f3460', label='Others')
], loc='lower right')
plt.tight_layout()
plt.show()""")

md("### 4b. xG vs Actual Goals")

code("""# xG vs Actual Goals scatter plot
if xg_col and 'goals_for' in master_df.columns:
    fig, ax = plt.subplots(figsize=(12, 10))

    x = master_df[xg_col]
    y = master_df['goals_for']

    ax.scatter(x, y, s=150, c='#e94560', edgecolors='white', linewidth=1.5, zorder=5)

    # Add team labels
    for _, row in master_df.iterrows():
        ax.annotate(row['team'], (row[xg_col], row['goals_for']),
                   textcoords="offset points", xytext=(8, 5),
                   fontsize=9, color='white', fontweight='bold')

    # Diagonal line (xG = Goals)
    lims = [min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2]
    ax.plot(lims, lims, '--', color='#e94560', alpha=0.5, label='xG = Actual Goals')

    ax.set_xlabel(f'{xg_col} (Expected Goals)', fontweight='bold')
    ax.set_ylabel('Actual Goals Scored', fontweight='bold')
    ax.set_title('xG vs Actual Goals - Who\\'s Over/Under Performing?', fontweight='bold', fontsize=16, color='#e94560')
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("xG data not available for scatter plot")
    if 'goals_for' in master_df.columns and 'goals_against' in master_df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(master_df['goals_for'], master_df['goals_against'], s=150, c='#e94560', edgecolors='white')
        for _, row in master_df.iterrows():
            ax.annotate(row['team'], (row['goals_for'], row['goals_against']),
                       textcoords="offset points", xytext=(8, 5), fontsize=9, color='white')
        ax.set_xlabel('Goals Scored', fontweight='bold')
        ax.set_ylabel('Goals Conceded', fontweight='bold')
        ax.set_title('Attack vs Defense', fontweight='bold', fontsize=16, color='#e94560')
        plt.tight_layout()
        plt.show()""")

md("### 4c. Attack vs Defense Quadrant Chart")

code("""if 'goals_per_match' in master_df.columns and 'goals_conceded_per_match' in master_df.columns:
    fig, ax = plt.subplots(figsize=(12, 10))

    x = master_df['goals_per_match']
    y = master_df['goals_conceded_per_match']

    ax.scatter(x, y, s=180, c=master_df['pts'], cmap='RdYlGn', edgecolors='white', linewidth=1.5, zorder=5)

    # Add quadrant lines at mean
    ax.axvline(x.mean(), color='#e94560', linestyle='--', alpha=0.5)
    ax.axhline(y.mean(), color='#e94560', linestyle='--', alpha=0.5)

    # Labels
    for _, row in master_df.iterrows():
        ax.annotate(row['team'], (row['goals_per_match'], row['goals_conceded_per_match']),
                   textcoords="offset points", xytext=(8, 5),
                   fontsize=9, color='white', fontweight='bold')

    # Quadrant labels
    ax.text(x.max(), y.min(), 'ELITE\\n(Score lots, concede few)', ha='right', va='bottom',
            fontsize=10, color='#00ff88', alpha=0.7, fontweight='bold')
    ax.text(x.min(), y.max(), 'STRUGGLING\\n(Score few, concede lots)', ha='left', va='top',
            fontsize=10, color='#ff4444', alpha=0.7, fontweight='bold')

    ax.set_xlabel('Goals Scored per Match', fontweight='bold')
    ax.set_ylabel('Goals Conceded per Match', fontweight='bold')
    ax.set_title('Attack vs Defense Quadrant', fontweight='bold', fontsize=16, color='#e94560')
    ax.invert_yaxis()  # Lower conceded = better = top
    plt.colorbar(ax.collections[0], label='Points')
    plt.tight_layout()
    plt.show()""")

md("### 4d. Key Stats Heatmap (Top 8 Teams)")

code("""# Select top 8 teams and key numeric columns for heatmap
top8 = master_df.nlargest(8, 'pts')
numeric_cols = [c for c in master_df.columns if master_df[c].dtype in ['float64', 'int64']
                and c not in ['played', 'hist_seasons_found', 'form_matches']]

# Limit to most interesting columns (max 15)
if len(numeric_cols) > 15:
    # Prioritize: pts, goals, xg, rates, form
    priority_keywords = ['pts', 'goal', 'xg', 'win', 'rate', 'form', 'shot', 'pass', 'tackle', 'possess']
    scored = []
    for col in numeric_cols:
        score = sum(1 for kw in priority_keywords if kw in col.lower())
        scored.append((col, score))
    scored.sort(key=lambda x: -x[1])
    numeric_cols = [c for c, s in scored[:15]]

if len(numeric_cols) > 0:
    heatmap_data = top8.set_index('team')[numeric_cols]

    # Normalize each column 0-1
    heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-10)

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(heatmap_norm, annot=heatmap_data.round(1).values, fmt='', cmap='RdYlGn',
                linewidths=1, linecolor='#1a1a2e', ax=ax, cbar_kws={'label': 'Normalized Score'})
    ax.set_title('Top 8 Teams - Key Stats Comparison', fontweight='bold', fontsize=16, color='#e94560')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()""")

# ============================================================
# SECTION 5: ML MODEL TRAINING
# ============================================================
md("---\n## 5. ML Model Training")

md("### 5a. Prepare Features & Target")

code("""# Target: points per match (we'll project to full season)
target = 'pts_per_match'

# Select features - exclude identifiers, target, and direct same-season outcome/leakage columns
exclude_cols = [
    'team', 'pts', 'pts_per_match', 'played', 'wins', 'draws', 'losses',
    'win_rate', 'draw_rate', 'loss_rate', 'goal_diff', 'goals_for', 'goals_against',
    'goals_per_match', 'goals_conceded_per_match', 'form_points_last5', 'form_matches',
    'hist_seasons_found'
]
feature_cols = [c for c in master_df.columns if c not in exclude_cols
                and master_df[c].dtype in ['float64', 'int64']
                and master_df[c].notna().sum() > len(master_df) * 0.5]  # At least 50% non-null

print(f"Target: {target}")
print(f"Features ({len(feature_cols)}):")
for f in feature_cols:
    print(f"  - {f}")

# Prepare X and y
X = master_df[feature_cols].fillna(0)
y = master_df[target]

print(f"\\nX shape: {X.shape}")
print(f"y shape: {y.shape}")""")

md("### 5b. Train & Evaluate Models (Leave-One-Out CV)")

code("""# Models to compare
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0),
    'Linear Regression': LinearRegression()
}

# Leave-One-Out CV (perfect for small datasets like 18 teams)
loo = LeaveOneOut()
results = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_absolute_error')
    mae = -scores.mean()
    results[name] = {
        'MAE': mae,
        'Std': scores.std(),
        'model': model
    }
    print(f"{name}: MAE = {mae:.4f} (+/- {scores.std():.4f})")

# Best model
best_model_name = min(results, key=lambda k: results[k]['MAE'])
print(f"\\nBest model: {best_model_name} (MAE = {results[best_model_name]['MAE']:.4f})")""")

md("### 5c. Train Best Model on Full Data & Feature Importance")

code("""# Train best model on full dataset
best_model = results[best_model_name]['model']
best_model.fit(X, y)

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_)
else:
    importances = np.zeros(len(feature_cols))

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=True)

# Plot feature importance
fig, ax = plt.subplots(figsize=(12, max(8, len(feature_cols) * 0.4)))
colors = ['#e94560' if 'hist' in f else '#0f3460' for f in importance_df['feature']]
ax.barh(importance_df['feature'], importance_df['importance'], color=colors, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Feature Importance', fontweight='bold')
ax.set_title(f'Feature Importance ({best_model_name})', fontweight='bold', fontsize=16, color='#e94560')
ax.legend(handles=[
    mpatches.Patch(color='#e94560', label='Historical features'),
    mpatches.Patch(color='#0f3460', label='Current season features')
], loc='lower right')
plt.tight_layout()
plt.show()

# Check historical feature importance
hist_importance = importance_df[importance_df['feature'].str.contains('hist')]['importance'].sum()
total_importance = importance_df['importance'].sum()
hist_pct = (hist_importance / total_importance * 100) if total_importance > 0 else 0.0
print(f"\\nHistorical features contribute {hist_pct:.1f}% of total importance")
print("(Goal: historical should be LOW, current season should dominate)")""")

# ============================================================
# SECTION 6: MONTE CARLO SIMULATION
# ============================================================
md("---\n## 6. Monte Carlo Simulation")

code("""# Predict points per match for each team
master_df['predicted_ppg'] = best_model.predict(X)

# Also train all models for comparison
for name, result in results.items():
    model = result['model']
    model.fit(X, y)
    master_df[f'pred_{name.lower().replace(" ", "_")}'] = model.predict(X)

# Total matches in Saudi Pro League season (18 teams, each plays 34 matches)
TOTAL_MATCHES = 34
remaining_matches = TOTAL_MATCHES - master_df['played'].values

print("Predicted Points Per Game vs Actual:")
compare = master_df[['team', 'pts_per_match', 'predicted_ppg', 'played']].copy()
compare['projected_pts'] = master_df['pts'] + (master_df['predicted_ppg'] * remaining_matches)
compare = compare.sort_values('projected_pts', ascending=False)
print(compare.to_string(index=False))""")

code("""# Monte Carlo Simulation - 10,000 runs
np.random.seed(42)
N_SIMULATIONS = 10000

# Team strength = predicted ppg with some noise
team_strengths = master_df.set_index('team')['predicted_ppg']
current_points = master_df.set_index('team')['pts']
team_remaining = dict(zip(master_df['team'], remaining_matches))

championship_wins = {team: 0 for team in master_df['team']}
final_points_all = {team: [] for team in master_df['team']}

for sim in range(N_SIMULATIONS):
    simulated_points = current_points.copy()

    for team in master_df['team']:
        rem = team_remaining[team]
        if rem > 0:
            ppg = team_strengths[team]
            # Add noise: each match result has variance
            # Simulate match-by-match: each match gives 0, 1, or 3 points
            noise_ppg = ppg + np.random.normal(0, 0.15)
            noise_ppg = max(0, min(3, noise_ppg))

            # Simulate remaining matches
            match_points = 0
            for _ in range(int(rem)):
                rand = np.random.random()
                # Convert ppg to win/draw/loss probabilities
                win_prob = noise_ppg / 3 * 0.85
                draw_prob = (1 - win_prob) * 0.4

                if rand < win_prob:
                    match_points += 3
                elif rand < win_prob + draw_prob:
                    match_points += 1

            simulated_points[team] += match_points

    # Record
    # Resolve ties fairly instead of always picking the first index.
    top_points = simulated_points.max()
    tied = simulated_points[simulated_points == top_points].index.tolist()
    champion = np.random.choice(tied)
    championship_wins[champion] += 1
    for team in master_df['team']:
        final_points_all[team].append(simulated_points[team])

# Calculate probabilities
championship_prob = {team: wins/N_SIMULATIONS*100 for team, wins in championship_wins.items()}
championship_prob = dict(sorted(championship_prob.items(), key=lambda x: -x[1]))

print(f"Championship Probabilities ({N_SIMULATIONS:,} simulations):")
print("-" * 45)
for team, prob in championship_prob.items():
    if prob > 0:
        avg_pts = np.mean(final_points_all[team])
        std_pts = np.std(final_points_all[team])
        print(f"  {team:25s}: {prob:5.1f}%  (Avg pts: {avg_pts:.0f} +/- {std_pts:.1f})")""")

# ============================================================
# SECTION 7: PREDICTION & FINAL RESULTS
# ============================================================
md("---\n## 7. Prediction & Final Results")

md("### THE BIG REVEAL")

code("""# Final Predicted Standings
master_df['projected_final_pts'] = master_df['pts'] + (master_df['predicted_ppg'] * remaining_matches)
master_df['avg_simulated_pts'] = [np.mean(final_points_all[t]) for t in master_df['team']]
master_df['std_simulated_pts'] = [np.std(final_points_all[t]) for t in master_df['team']]
master_df['championship_prob'] = [championship_prob.get(t, 0) for t in master_df['team']]

# Sort by championship probability
final_standings = master_df[['team', 'pts', 'played', 'predicted_ppg', 'projected_final_pts',
                              'avg_simulated_pts', 'std_simulated_pts', 'championship_prob']].copy()
final_standings = final_standings.sort_values('championship_prob', ascending=False)

predicted_champion = final_standings.iloc[0]['team']
champion_prob = final_standings.iloc[0]['championship_prob']

print("=" * 70)
print(f"  PREDICTED CHAMPION: {predicted_champion}")
print(f"  Championship Probability: {champion_prob:.1f}%")
print(f"  Projected Points: {final_standings.iloc[0]['avg_simulated_pts']:.0f}")
print("=" * 70)

print(f"\\n{'Rank':<5} {'Team':<25} {'Current Pts':<13} {'Proj. Pts':<12} {'Win Prob %':<10}")
print("-" * 65)
for i, (_, row) in enumerate(final_standings.iterrows()):
    print(f"{i+1:<5} {row['team']:<25} {int(row['pts']):<13} {row['avg_simulated_pts']:<12.0f} {row['championship_prob']:<10.1f}")""")

# ============================================================
# SECTION 8: FINAL DASHBOARD
# ============================================================
md("---\n## 8. Final Visualization Dashboard")

md("### 8a. Championship Probability")

code("""# Championship probability bar chart (THE HERO VISUAL)
prob_df = final_standings[final_standings['championship_prob'] > 0].sort_values('championship_prob', ascending=True)

if len(prob_df) == 0:
    prob_df = final_standings.nlargest(5, 'projected_final_pts')
    prob_df = prob_df.sort_values('projected_final_pts', ascending=True)

fig, ax = plt.subplots(figsize=(14, max(6, len(prob_df) * 0.8)))

gradient_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(prob_df)))
bars = ax.barh(prob_df['team'], prob_df['championship_prob'], color=gradient_colors,
               edgecolor='white', linewidth=1)

for bar, prob in zip(bars, prob_df['championship_prob']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{prob:.1f}%', va='center', fontweight='bold', color='white', fontsize=13)

ax.set_xlabel('Championship Probability (%)', fontweight='bold', fontsize=14)
ax.set_title(f'Who Will Win the Saudi Pro League 2024/25?\\n({N_SIMULATIONS:,} Monte Carlo Simulations)',
             fontweight='bold', fontsize=18, color='#e94560')
plt.tight_layout()
plt.show()""")

md("### 8b. Predicted Final Points with Error Bars")

code("""# Predicted points with confidence intervals
sorted_final = final_standings.sort_values('avg_simulated_pts', ascending=True)

fig, ax = plt.subplots(figsize=(14, 10))

colors = ['#e94560' if i >= len(sorted_final) - 3 else '#0f3460' for i in range(len(sorted_final))]

ax.barh(sorted_final['team'], sorted_final['avg_simulated_pts'],
        xerr=sorted_final['std_simulated_pts'], color=colors,
        edgecolor='white', linewidth=0.5, capsize=3, error_kw={'color': 'white', 'linewidth': 1.5})

# Mark current points
ax.scatter(sorted_final['pts'], sorted_final['team'], color='#00ff88', s=80, zorder=5,
           label='Current Points', marker='D')

for _, row in sorted_final.iterrows():
    ax.text(row['avg_simulated_pts'] + row['std_simulated_pts'] + 1,
            row['team'], f"{row['avg_simulated_pts']:.0f}",
            va='center', fontweight='bold', color='white', fontsize=10)

ax.set_xlabel('Points', fontweight='bold')
ax.set_title('Predicted Final Points (with uncertainty)', fontweight='bold', fontsize=16, color='#e94560')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()""")

md("### 8c. Top 3 Contenders Radar Chart")

code("""# Radar chart for top 3 contenders
top3 = final_standings.nlargest(3, 'championship_prob')
radar_cols = ['win_rate', 'goals_per_match', 'pts_per_match']

# Add more radar cols if available
for col in ['goals_conceded_per_match', 'form_ppg']:
    if col in master_df.columns:
        radar_cols.append(col)

# Add any xG columns
if xg_col:
    radar_cols.append(xg_col)
if xga_col:
    radar_cols.append(xga_col)

# Ensure we have at least 4 dimensions
available_radar = [c for c in radar_cols if c in master_df.columns and master_df[c].notna().all()]
if len(available_radar) < 4:
    for c in master_df.select_dtypes(include=[np.number]).columns:
        if c not in available_radar and c not in ['played', 'pts', 'pts_per_match'] and len(available_radar) < 6:
            available_radar.append(c)

if len(available_radar) >= 3:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(available_radar), endpoint=False).tolist()
    angles += angles[:1]

    colors_radar = ['#e94560', '#00ff88', '#ffd700']

    for i, (_, row) in enumerate(top3.iterrows()):
        team_name = row['team']
        team_data = master_df[master_df['team'] == team_name]

        values = []
        for col in available_radar:
            val = team_data[col].values[0]
            # Normalize 0-1 against all teams
            col_min = master_df[col].min()
            col_max = master_df[col].max()
            if col_max > col_min:
                # For conceded stats, invert (lower is better)
                if 'conceded' in col.lower() or 'against' in col.lower() or 'xga' in col.lower():
                    val = 1 - (val - col_min) / (col_max - col_min)
                else:
                    val = (val - col_min) / (col_max - col_min)
            else:
                val = 0.5
            values.append(val)
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=team_name, color=colors_radar[i])
        ax.fill(angles, values, alpha=0.15, color=colors_radar[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace('_', ' ').title() for c in available_radar], fontsize=9)
    ax.set_title('Top 3 Contenders Comparison', fontweight='bold', fontsize=16, color='#e94560', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.show()""")

md("### 8d. Model Comparison")

code("""# Model comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))

model_names = list(results.keys())
maes = [results[m]['MAE'] for m in model_names]
colors = ['#e94560' if m == best_model_name else '#0f3460' for m in model_names]

bars = ax.bar(model_names, maes, color=colors, edgecolor='white', linewidth=1)

for bar, mae in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{mae:.4f}', ha='center', fontweight='bold', color='white')

ax.set_ylabel('Mean Absolute Error (lower = better)', fontweight='bold')
ax.set_title('Model Comparison (Leave-One-Out CV)', fontweight='bold', fontsize=16, color='#e94560')
ax.legend(handles=[
    mpatches.Patch(color='#e94560', label=f'Best: {best_model_name}'),
    mpatches.Patch(color='#0f3460', label='Other models')
])
plt.tight_layout()
plt.show()""")

md("### 8e. Summary")

code("""# Final summary
print("=" * 70)
print("  SAUDI PRO LEAGUE 2024/25 - PREDICTION SUMMARY")
print("=" * 70)
print(f"\\n  Data Source: FotMob API")
print(f"  Stats Used: {len(feature_cols)} features across {len(all_stats)} stat categories")
print(f"  Historical Seasons: {len(historical_standings)} seasons analyzed")
print(f"  Best ML Model: {best_model_name} (MAE: {results[best_model_name]['MAE']:.4f})")
print(f"  Simulations: {N_SIMULATIONS:,} Monte Carlo runs")
print(f"\\n  PREDICTED CHAMPION: {predicted_champion}")
print(f"  Win Probability: {champion_prob:.1f}%")
print(f"  Projected Points: {final_standings.iloc[0]['avg_simulated_pts']:.0f} +/- {final_standings.iloc[0]['std_simulated_pts']:.1f}")
print(f"\\n  Top 3 Contenders:")
for i, (_, row) in enumerate(final_standings.head(3).iterrows()):
    print(f"    {i+1}. {row['team']} - {row['championship_prob']:.1f}% chance ({row['avg_simulated_pts']:.0f} pts)")
print(f"\\n  Historical features weight: {hist_pct:.1f}% (target: LOW)")
print("=" * 70)
print("\\n  Model by: AI-Powered Saudi League Analysis")
print("  Subscribe for more football predictions!")""")

# ============================================================
# FIX CELL SOURCES: join lines properly for .ipynb format
# ============================================================
for cell in cells:
    # Rejoin and re-split to add newlines properly
    full_source = "\n".join(cell["source"])
    lines = full_source.split("\n")
    # Each line except the last needs a trailing \n
    cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": cells
}

output_path = Path(__file__).resolve().parent / "saudi_league_prediction.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to {output_path}")
print(f"Total cells: {len(cells)}")
print(f"Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
print(f"Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
