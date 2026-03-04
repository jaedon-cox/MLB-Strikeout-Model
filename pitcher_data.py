#!/usr/bin/env python3
"""
Pitcher strikeout data collector.
"""

# =============================================================================
# PARAMETERS — adjust these before running
# =============================================================================

START_DATE = '2021-04-01'   # Format: 'YYYY-MM-DD'
END_DATE   = '2021-05-01'   # Format: 'YYYY-MM-DD'

# =============================================================================

import pandas as pd
import numpy as np
import statsapi
from datetime import datetime, timedelta
from collections import defaultdict
from pybaseball import statcast_pitcher


def get_all_games(start_date, end_date):
    """
    Returns all final MLB games between start_date and end_date.

    Args:
        start_date: String 'YYYY-MM-DD'
        end_date:   String 'YYYY-MM-DD'

    Returns:
        List of dicts, one per game.
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end   = datetime.strptime(end_date,   '%Y-%m-%d')

    all_games = []
    current_date = start

    while current_date <= end:
        date_str = current_date.strftime('%m/%d/%Y')
        try:
            schedule = statsapi.schedule(date=date_str)
            final_games = [g for g in schedule if g['status'] == 'Final']

            for game in final_games:
                all_games.append({
                    'game_id':   game['game_id'],
                    'game_date': current_date.strftime('%Y-%m-%d'),
                    'home_team': game['home_name'],
                    'away_team': game['away_name'],
                    'home_id':   game['home_id'],
                    'away_id':   game['away_id']
                })

            print(f"  {date_str}: {len(final_games)} final games")

        except Exception as e:
            print(f"  Error on {date_str}: {e}")

        current_date += timedelta(days=1)

    print(f"\nTotal games found: {len(all_games)}")
    return all_games


def get_starting_pitchers(game_id, boxscore):
    """
    Returns both starting pitchers and their game stats for a given game.

    Args:
        game_id:  MLB game ID (from get_all_games)
        boxscore: Already-fetched boxscore dict (from statsapi.boxscore_data)

    Returns:
        List of 2 dicts (home and away starter), or empty list on failure.
    """
    try:
        pitchers = []

        for side, opp in [('home', 'away'), ('away', 'home')]:
            side_data   = boxscore.get(side, {})
            pitcher_ids = side_data.get('pitchers', [])
            players     = side_data.get('players', {})

            if not pitcher_ids:
                continue

            starter_id  = pitcher_ids[0]
            pitcher_key = f'ID{starter_id}'

            if pitcher_key not in players:
                continue

            info      = players[pitcher_key]
            person    = info.get('person', {})
            stats     = info.get('stats', {}).get('pitching', {})
            team_info = boxscore.get('teamInfo', {})

            pitchers.append({
                'pitcher_id':      starter_id,
                'pitcher_name':    person.get('fullName', 'Unknown'),
                'team_id':         team_info.get(side, {}).get('id'),
                'team_name':       team_info.get(side, {}).get('teamName'),
                'opponent_id':     team_info.get(opp, {}).get('id'),
                'opponent_name':   team_info.get(opp, {}).get('teamName'),
                'strikeouts':      stats.get('strikeOuts', 0),
                'pitches_thrown':  stats.get('numberOfPitches', 0),
                'innings_pitched': stats.get('inningsPitched', '0.0')
            })

        return pitchers

    except Exception as e:
        print(f"  Error getting pitchers for game {game_id}: {e}")
        return []


def get_statcast_data(pitcher_id, end_date):
    """
    Fetches Statcast pitch-by-pitch data for a pitcher over the 60 days
    leading up to (but not including) end_date.

    Args:
        pitcher_id: MLB pitcher ID
        end_date:   String 'YYYY-MM-DD' — pulls data BEFORE this date

    Returns:
        DataFrame of pitch data, or empty DataFrame on failure.
    """
    try:
        end   = datetime.strptime(end_date, '%Y-%m-%d')
        start = end - timedelta(days=60)

        data = statcast_pitcher(
            start.strftime('%Y-%m-%d'),
            end_date,
            pitcher_id
        )

        if data is not None and not data.empty:
            return data

    except Exception as e:
        print(f"  Statcast error for pitcher {pitcher_id}: {e}")

    return pd.DataFrame()


def calculate_pitch_metrics(pitch_data):
    """
    Calculates advanced pitch metrics from a Statcast DataFrame.

    Args:
        pitch_data: DataFrame returned by get_statcast_data()

    Returns:
        Dict of calculated metrics, or NaN-filled dict if no data.
    """
    empty = {
        'swstr_pct':             np.nan,
        'whiff_pct':             np.nan,
        'chase_pct':             np.nan,
        'csw_pct':               np.nan,
        'first_pitch_strike_pct': np.nan,
        'avg_velo':              np.nan,
        'avg_spin':              np.nan,
        'avg_h_break':           np.nan,
        'avg_v_break':           np.nan,
    }

    if pitch_data.empty:
        return empty

    total = len(pitch_data)

    # Swinging strikes
    swinging_strikes = pitch_data['description'].isin([
        'swinging_strike', 'swinging_strike_blocked'
    ]).sum()

    # All swings (swinging strikes + fouls)
    swings = pitch_data['description'].str.contains(
        'swinging|foul', case=False, na=False
    ).sum()

    # Called strikes
    called_strikes = pitch_data['description'].isin(['called_strike']).sum()

    # Pitches outside the zone (zones 11-14)
    outside_zone = (
        pitch_data['zone'].isin([11, 12, 13, 14]).sum()
        if 'zone' in pitch_data.columns else 0
    )

    # Chase swings (swings at pitches outside zone)
    chase_swings = 0
    if 'zone' in pitch_data.columns and outside_zone > 0:
        outside_pitches = pitch_data[pitch_data['zone'].isin([11, 12, 13, 14])]
        chase_swings = outside_pitches['description'].str.contains(
            'swinging|foul', case=False, na=False
        ).sum()

    # First pitch strikes
    first_pitches = (
        pitch_data[pitch_data['pitch_number'] == 1]
        if 'pitch_number' in pitch_data.columns else pd.DataFrame()
    )
    first_pitch_strikes = 0
    if not first_pitches.empty:
        first_pitch_strikes = first_pitches['description'].isin([
            'called_strike', 'swinging_strike', 'swinging_strike_blocked', 'foul'
        ]).sum()

    return {
        'swstr_pct':              round(swinging_strikes / total * 100, 2)       if total > 0        else np.nan,
        'whiff_pct':              round(swinging_strikes / swings * 100, 2)      if swings > 0       else np.nan,
        'chase_pct':              round(chase_swings / outside_zone * 100, 2)    if outside_zone > 0 else np.nan,
        'csw_pct':                round((called_strikes + swinging_strikes) / total * 100, 2) if total > 0 else np.nan,
        'first_pitch_strike_pct': round(first_pitch_strikes / len(first_pitches) * 100, 2)   if len(first_pitches) > 0 else np.nan,
        'avg_velo':               round(pitch_data['release_speed'].mean(), 1)   if 'release_speed'      in pitch_data.columns else np.nan,
        'avg_spin':               round(pitch_data['release_spin_rate'].mean(), 0) if 'release_spin_rate' in pitch_data.columns else np.nan,
        'avg_h_break':            round(pitch_data['pfx_x'].mean(), 2)           if 'pfx_x'              in pitch_data.columns else np.nan,
        'avg_v_break':            round(pitch_data['pfx_z'].mean(), 2)           if 'pfx_z'              in pitch_data.columns else np.nan,
    }


def calculate_rolling_stats(pitcher_history, n_games=5):
    """
    Calculates rolling averages from a pitcher's last N starts.
    If fewer than N games are available, uses whatever is available.

    Args:
        pitcher_history: List of dicts, one per prior start, each containing:
                         'date', 'strikeouts', 'pitches_thrown',
                         'k_pct', 'swstr_pct', 'whiff_pct'
        n_games:         Number of recent starts to average over (default 5)

    Returns:
        Dict of rolling stats.
    """
    empty = {
        'rolling_k_pct':     np.nan,
        'rolling_swstr':     np.nan,
        'rolling_whiff':     np.nan,
        'rolling_pitches':   np.nan,
        'rest_days':         np.nan,
    }

    if len(pitcher_history) == 0:
        return empty

    # Use last N games, or all available if fewer than N
    recent = pitcher_history[-n_games:]

    # Rest days = days between the two most recent starts
    rest_days = np.nan
    if len(pitcher_history) >= 2:
        last_start = pitcher_history[-1]['date']
        prev_start = pitcher_history[-2]['date']
        rest_days  = (last_start - prev_start).days

    # Pull each metric, skipping NaN values
    def avg(key):
        vals = [g[key] for g in recent if not pd.isna(g.get(key, np.nan))]
        return round(float(np.mean(vals)), 2) if vals else np.nan

    return {
        'rolling_k_pct':   avg('k_pct'),
        'rolling_swstr':   avg('swstr_pct'),
        'rolling_whiff':   avg('whiff_pct'),
        'rolling_pitches': avg('pitches_thrown'),
        'rest_days':       rest_days,
    }


def get_team_k_rate(team_id):
    """
    Returns the season-to-date batting K rate for a team based on
    games already processed. Returns NaN if no data yet.
    """
    stats = team_batting[team_id]
    pa    = stats['plate_appearances']
    if pa == 0:
        return np.nan
    return round(stats['strikeouts'] / pa * 100, 2)


def update_team_batting(boxscore):
    """
    Pulls batting strikeouts and plate appearances from a boxscore
    and adds them to both teams' running totals.
    """
    for side in ['home', 'away']:
        side_data = boxscore.get(side, {})
        team_id   = boxscore.get('teamInfo', {}).get(side, {}).get('id')
        if not team_id:
            continue

        # Sum batting stats across all batters in this game
        players = side_data.get('players', {})
        for player_key, player_data in players.items():
            batting = player_data.get('stats', {}).get('batting', {})
            team_batting[team_id]['strikeouts']       += batting.get('strikeOuts', 0)
            team_batting[team_id]['plate_appearances'] += batting.get('plateAppearances', 0)


# =============================================================================
# MAIN — collect and combine all data
# =============================================================================

CHECKPOINT_EVERY = 100       # Save a checkpoint every N games
CHECKPOINT_FILE  = 'pitcher_data_checkpoint.csv'

print(f"Fetching games from {START_DATE} to {END_DATE}...")
games = get_all_games(START_DATE, END_DATE)

pitcher_histories = defaultdict(list)
all_data = []

# Track each team's cumulative batting strikeouts and plate appearances
# { team_id: { 'strikeouts': int, 'plate_appearances': int } }
team_batting = defaultdict(lambda: {'strikeouts': 0, 'plate_appearances': 0})

for i, game in enumerate(games):
    print(f"\nGame {i + 1}/{len(games)}: {game['game_date']} — {game['away_team']} @ {game['home_team']}")

    # Fetch boxscore once and share it across pitcher processing + team batting update
    try:
        boxscore = statsapi.boxscore_data(game['game_id'])
    except Exception as e:
        print(f"  Error fetching boxscore: {e}")
        boxscore = None

    pitchers = get_starting_pitchers(game['game_id'], boxscore) if boxscore else []

    for pitcher in pitchers:
        pitcher_id  = pitcher['pitcher_id']
        game_date   = datetime.strptime(game['game_date'], '%Y-%m-%d')
        opponent_id = pitcher['opponent_id']

        # Only use history from BEFORE this game
        prior_history = [g for g in pitcher_histories[pitcher_id] if g['date'] < game_date]

        # Rolling stats from prior starts
        rolling = calculate_rolling_stats(prior_history)

        # Statcast data up to the day before this game
        day_before    = (game_date - timedelta(days=1)).strftime('%Y-%m-%d')
        pitch_data    = get_statcast_data(pitcher_id, day_before)
        pitch_metrics = calculate_pitch_metrics(pitch_data)

        # Opponent K rate from games processed so far (before updating with this game)
        opponent_k_rate = get_team_k_rate(opponent_id)

        # Convert innings pitched from MLB format (6.1 = 6⅓, 6.2 = 6⅔) to true decimal
        raw_ip = pitcher['innings_pitched']
        try:
            whole, thirds = str(raw_ip).split('.')
            innings_pitched_decimal = int(whole) + int(thirds) / 3
            innings_pitched_decimal = round(innings_pitched_decimal, 4)
        except Exception:
            innings_pitched_decimal = np.nan

        # Home/away flag — 1 if pitching at home, 0 if away
        is_home = 1 if pitcher['team_id'] == game['home_id'] else 0

        # Build the record for this pitcher-game
        record = {
            'game_id':          game['game_id'],
            'game_date':        game['game_date'],
            'pitcher_id':       pitcher_id,
            'pitcher_name':     pitcher['pitcher_name'],
            'team':             pitcher['team_name'],
            'opponent':         pitcher['opponent_name'],
            'is_home':          is_home,
            'strikeouts':       pitcher['strikeouts'],
            'pitches_thrown':   pitcher['pitches_thrown'],
            'innings_pitched':  innings_pitched_decimal,
            'opponent_k_rate':  opponent_k_rate,
            **pitch_metrics,
            **rolling,
        }

        all_data.append(record)

        # Update this pitcher's history for future games
        pitcher_histories[pitcher_id].append({
            'date':           game_date,
            'strikeouts':     pitcher['strikeouts'],
            'pitches_thrown': pitcher['pitches_thrown'],
            'k_pct':          (pitcher['strikeouts'] / pitcher['pitches_thrown'] * 100)
                               if pitcher['pitches_thrown'] > 0 else np.nan,
            'swstr_pct':      pitch_metrics.get('swstr_pct', np.nan),
            'whiff_pct':      pitch_metrics.get('whiff_pct', np.nan),
        })

    # Update team batting stats AFTER building pitcher records for this game
    # so opponent K rate reflects data BEFORE this game
    if boxscore:
        update_team_batting(boxscore)

    # Checkpoint every N games
    if (i + 1) % CHECKPOINT_EVERY == 0:
        pd.DataFrame(all_data).to_csv(CHECKPOINT_FILE, index=False)
        print(f"\n  ✓ Checkpoint saved at game {i + 1} → {CHECKPOINT_FILE}")

df = pd.DataFrame(all_data)

print(f"\n{'=' * 60}")
print(f"COLLECTION COMPLETE")
print(f"{'=' * 60}")
print(f"Total records : {len(df)}")
print(f"Unique pitchers: {df['pitcher_id'].nunique()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")


# =============================================================================
# OFFSET DATA FOR PREDICTION
# =============================================================================
# Each row will contain features from start N predicting strikeouts from start N+1

df = df.sort_values(['pitcher_id', 'game_date']).reset_index(drop=True)

# Columns we don't want to use as features
exclude_cols = ['strikeouts', 'game_id', 'game_date', 'pitcher_id',
                'pitcher_name', 'team', 'opponent']

feature_cols = [col for col in df.columns if col not in exclude_cols]
info_cols    = ['game_id', 'game_date', 'pitcher_id', 'pitcher_name', 'team', 'opponent']

offset_rows = []

for pitcher_id in df['pitcher_id'].unique():
    pitcher_games = df[df['pitcher_id'] == pitcher_id].reset_index(drop=True)

    # Need at least 2 starts — one for features, one for the target
    if len(pitcher_games) < 2:
        continue

    for i in range(1, len(pitcher_games)):
        prev = pitcher_games.iloc[i - 1]   # features come from this start
        curr = pitcher_games.iloc[i]        # target (strikeouts) comes from this start

        row = {}

        # Identifying info from the CURRENT game (the one being predicted)
        for col in info_cols:
            row[col] = curr[col]

        # All features from the PREVIOUS game (including pitches_thrown and innings_pitched)
        # We won't know these values for the current game at prediction time
        for col in feature_cols:
            row[f'prev_{col}'] = prev[col]

        # Target from the CURRENT game only
        row['strikeouts'] = curr['strikeouts']

        offset_rows.append(row)

df_offset = pd.DataFrame(offset_rows)

print(f"\n{'=' * 60}")
print(f"OFFSET COMPLETE")
print(f"{'=' * 60}")
print(f"Original rows : {len(df)}")
print(f"Offset rows   : {len(df_offset)}  (one start dropped per pitcher)")
print(f"Feature columns: {[c for c in df_offset.columns if c.startswith('prev_')]}")


# =============================================================================
# DROP INCOMPLETE ROWS
# =============================================================================
# Remove any rows missing feature values (e.g. no prior game history)

prev_cols = [col for col in df_offset.columns if col.startswith('prev_')]

before = len(df_offset)
df_clean = df_offset.dropna(subset=prev_cols).reset_index(drop=True)
after = len(df_clean)

print(f"\n{'=' * 60}")
print(f"CLEANING COMPLETE")
print(f"{'=' * 60}")
print(f"Rows before : {before}")
print(f"Rows dropped: {before - after}  (incomplete feature data)")
print(f"Rows after  : {after}")
print(f"\nMissing values remaining:\n{df_clean.isnull().sum()[df_clean.isnull().sum() > 0]}")


# =============================================================================
# SAVE TO CSV
# =============================================================================

output_file = 'pitcher_training_data.csv'
df_clean.to_csv(output_file, index=False)

print(f"\n{'=' * 60}")
print(f"SAVED")
print(f"{'=' * 60}")
print(f"File    : {output_file}")
print(f"Rows    : {len(df_clean)}")
print(f"Columns : {len(df_clean.columns)}")
print(f"\nTarget variable (strikeouts):")
print(df_clean['strikeouts'].describe())
