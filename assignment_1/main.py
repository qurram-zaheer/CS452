import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import operator


sns.set_style("whitegrid", {'axes.grid': False})

raw_data = pd.read_excel('2018 ODI Circket matches.xlsx')


# -------------------------------- Question 1 -------------------------------- #

split_vals = raw_data['Margin'].str.split(" ", 1, expand=True)

raw_data['tm'] = split_vals[0]
raw_data['km'] = split_vals[1]

raw_data['Wickets'] = np.where(
    raw_data['km'] == 'wickets', raw_data['tm'], np.nan)
raw_data['Runs'] = np.where(raw_data['km'] == 'runs', raw_data['tm'], np.nan)

raw_data['Wickets'] = pd.to_numeric(raw_data['Wickets'])
raw_data['Runs'] = pd.to_numeric(raw_data['Runs'])

cleaned_data_v1 = raw_data.drop(['tm', 'km'], axis=1)

cleaned_data_v1.to_excel('phase1_cleaning.xlsx')


# -------------------------------- Question 2 -------------------------------- #

played_counts_1 = cleaned_data_v1['Team 1'].value_counts()
played_counts_2 = cleaned_data_v1['Team 2'].value_counts()

played_counts = played_counts_1.add(played_counts_2, fill_value=0)
most_played = played_counts.idxmax()

played_counts = played_counts.to_frame()
played_counts = played_counts.rename_axis('Country').reset_index()
played_counts = played_counts.rename(columns={0: 'Games Played'})

ax = sns.barplot(data=played_counts, x='Country', y='Games Played')
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='small'
)
plt.show()

# -------------------------------- Question 3 -------------------------------- #

win_counts = cleaned_data_v1['Winner'].value_counts()
win_counts = win_counts.drop(labels=['tied', 'no result'])
top_3 = win_counts.nlargest(n=3)
win_counts = win_counts.to_frame()
win_counts = win_counts.rename_axis('Country').reset_index()
win_counts = win_counts.rename(columns={0: 'Games Played'})

ax = sns.barplot(data=win_counts, x='Country', y='Winner')
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='small'
)
plt.show()

# -------------------------------- Question 4 -------------------------------- #

home_grounds = {
    'Scotland': ['Edinburgh'],
    'New Zealand': ['Wellington', 'Dunedin', 'Christchurch',
                    'Mount Maunganui', 'Hamilton', 'Nelson'],
    'Hong Kong': [],
    'U.A.E.': ['Abu Dhabi', 'Sharjah', 'ICCA Dubai', 'Dubai (DSC)'],
    'Zimbabwe': ['Bulawayo', 'Harare'],
    'West Indies': [],
    'Pakistan': [],
    'Afghanistan': [],
    'Nepal': [],
    'Bangladesh': ['Chattogram', 'Dhaka', 'Sylhet'],
    'England': ['Leeds', "Lord's", 'The Oval', 'Nottingham', 'Manchester', 'Cardiff'],
    'Netherlands': ['Amstelveen'],
    'South Africa': ['Bloemfontein', 'Port Elizabeth', 'Centurion', 'Paarl', 'Providence',
                     'Kimberley', 'Cape Town', 'Durban', 'Johannesburg'],
    'India': ['Pune', 'Thiruvananthapuram', 'Visakhapatnam', 'Mumbai (BS)', 'Guwahati'],
    'Ireland': ['Belfast'],
    'Australia': ['Melbourne', 'Adelaide', 'Brisbane', 'Hobart', 'Sydney', 'Perth'],
    'Sri Lanka': ['Kuala lumpur', 'Pallekele', 'Dambulla', 'Colombo (RPS)'],
    'P.N.G.': [],
}

teams_list = np.unique(np.append(
    pd.unique(cleaned_data_v1['Team 1']), pd.unique(cleaned_data_v1['Team 2'])))

home_games = {}

for team in teams_list:
    team_df = cleaned_data_v1.loc[((cleaned_data_v1['Team 1'] == team) | (
        cleaned_data_v1['Team 2'] == team)) & (cleaned_data_v1['Ground'].isin(home_grounds[team]))]
    print(team_df.shape[0])
    home_games[team] = team_df.shape[0]


maxhg = max(home_games.items(), key=operator.itemgetter(1))[1]
print(maxhg)
max_home_games = [key for key, value in home_games.items()
                  if value == maxhg]
# -------------------------------- Question 5 -------------------------------- #

sl_wr = win_counts.set_index('Country').loc['Sri Lanka']['Winner'] / \
    played_counts.set_index('Country').loc['Sri Lanka']['Games Played']

margins_df = cleaned_data_v1[['Winner', 'Runs', 'Wickets']].copy()
margins_df['Runs'] = pd.to_numeric(margins_df['Runs'])
margins_df['Wickets'] = pd.to_numeric(margins_df['Wickets'])
margins_df
grouped_margins = margins_df.groupby(['Winner']).mean()


sl_avg_runs_margin = grouped_margins.loc['Sri Lanka']['Runs']
sl_avg_wickets_margin = grouped_margins.loc['Sri Lanka']['Wickets']

sl_rank = np.where(win_counts['Country'] == 'Sri Lanka')
sl_wins_rank = sl_rank[0][0]

# -------------------------------- Question 6 -------------------------------- #

runs_descending = cleaned_data_v1.sort_values(
    'Runs', ascending=False, inplace=False)
top3_runmargins = runs_descending.iloc[0:3]

# -------------------------------- Question 7 -------------------------------- #

cleaned_data_v1['Month'] = cleaned_data_v1['Match Date'].dt.month
cleaned_data_v1['Month'] = cleaned_data_v1['Month'].transform(
    lambda x: calendar.month_name[x])

month_counts = cleaned_data_v1['Month'].value_counts()
most_played_month = month_counts.index[0]
# -------------------------------- Question 8 -------------------------------- #

away_counts = cleaned_data_v1['Team 1'].value_counts()
most_awaygames = away_counts.idxmax()

# -------------------------------- Question 9 -------------------------------- #


ax = sns.barplot(x=month_counts.index, y=month_counts.values)
plt.xticks(
    fontweight='light',
    fontsize='small'
)
plt.show()

# -------------------------------- Question 10 ------------------------------- #

ground_counts = cleaned_data_v1['Ground'].value_counts()
most_played_ground = ground_counts.index[0]

# -------------------------------- Question 11 ------------------------------- #

india_df = cleaned_data_v1.loc[cleaned_data_v1['Winner'] == 'India']

if india_df['Runs'].count() > india_df['Wickets'].count():
    india_most_wins = 'Batting first'
else:
    india_most_wins = 'Chasing'

# -------------------------------- Question 12 ------------------------------- #

top3_dfs = []

for index, country in enumerate(top_3.index):
    top3_dfs.append(cleaned_data_v1.loc[(cleaned_data_v1['Team 1'] == country) | (
        cleaned_data_v1['Team 2'] == country)])

month_wrs = {}

for index, country in enumerate(top3_dfs):
    t = top3_dfs[index]
    t3_wins = t.loc[t['Winner'] == top_3.index[index]]
    grouped_wins = t3_wins.groupby('Month').agg({'Winner': ['count']})
    grouped_played = t.groupby(
        'Month').agg({'Winner': ['count']})
    month_wrs[top_3.index[index]] = grouped_wins/grouped_played


# -------------------------------- Question 13 ------------------------------- #

losses = {}

for team in teams_list:
    team_df = cleaned_data_v1.loc[((cleaned_data_v1['Team 1'] == team) | (
        cleaned_data_v1['Team 2'] == team)) & (cleaned_data_v1['Winner'] != team)]
    print(team_df.shape[0])
    losses[team] = team_df.shape[0]

max_losses = max(losses.items(), key=operator.itemgetter(1))[1]
max_losses = [key for key, value in losses.items() if value == max_losses]
