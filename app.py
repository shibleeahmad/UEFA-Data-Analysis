import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Set Seaborn style
sns.set_style("whitegrid")

# Page setup
st.set_page_config(page_title="UEFA Champions League EDA", layout="wide")
st.title("⚽ UEFA Champions League - Exploratory Data Analysis")
st.markdown("---")

# Load datasets
df = pd.read_csv("performance.csv")
df1 = pd.read_csv("matches.csv")

# Data cleaning
df = df.drop(columns=['#'])
df['goal'] = df['goals'].str.split(':').str[0].astype(int)
df["Goals_per_Match"] = df["goal"] / df["M."]
df["Win_Loss_Ratio"] = df["W"] / df["L"]

df1[["Winner_Goals", "Runner_Goals"]] = df1["Score"].str.split("–", expand=True).astype(int)
df1["Score_Margin"] = abs(df1["Winner_Goals"] - df1["Runner_Goals"])

# Sidebar options
st.sidebar.header("Choose Analysis")
analysis = st.sidebar.radio("Select Data Type", ["Team Performance", "Finals History","ML Prediction"]) 

# --------------------- TEAM PERFORMANCE ---------------------
if analysis == "Team Performance":
    st.subheader("Team Performance Metrics")

    # Show only top 10 teams by matches played for all plots 
    top_teams = df.sort_values("M.", ascending=False).head(10)

    col1, col2 = st.columns(2)
    with col1:
        team = df.loc[df['M.'].idxmax(), 'Team']
        st.metric("🔝 Most Matches Played", team)

    with col2:
        top_goals_team = df.loc[df["Goals_per_Match"].idxmax(), "Team"]
        st.metric("⚽ Highest Goals per Match", top_goals_team)

    lowest_losses_team = df.loc[df["L"].idxmin(), "Team"]
    st.success(f"📉 Team with Lowest Losses: **{lowest_losses_team}**")

    st.markdown("### 1. Wins Comparison (Top 10 Teams)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_teams.sort_values("W", ascending=False), x="Team", y="W", palette="Blues_d", ax=ax1)
    ax1.set_title("Wins Among Top 10 Teams")
    ax1.set_xlabel("Team")
    ax1.set_ylabel("Wins") 
    ax1.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    st.pyplot(fig1)

    st.markdown("### 2. Total Matches vs Wins (Top 10 Teams)")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    width = 0.35
    x = range(len(top_teams))
    ax2.bar(x, top_teams["M."], width=width, label="Matches", color="skyblue")
    ax2.bar([i + width for i in x], top_teams["W"], width=width, label="Wins", color="navy")
    ax2.set_xticks([i + width/2 for i in x])
    ax2.set_xticklabels(top_teams["Team"], rotation=30)
    ax2.set_title("Total Matches vs Wins (Top 10 Teams)")
    ax2.set_xlabel("Team")
    ax2.set_ylabel("Count")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("### 3. Win/Loss Ratio (Top 10 Teams)")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_teams.sort_values("Win_Loss_Ratio", ascending=False), x="Team", y="Win_Loss_Ratio", palette="Purples_r", ax=ax3)
    ax3.set_title("Win/Loss Ratio (Top 10 Teams)")
    ax3.set_xlabel("Team")
    ax3.set_ylabel("Win/Loss Ratio")
    ax3.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("### 4. Goal Difference Heatmap (Top 10 Teams)")
    fig4, ax4 = plt.subplots(figsize=(8, 4)) 
    sns.heatmap(top_teams.set_index("Team")[["Dif"]], annot=True, cmap="coolwarm", linewidths=1, ax=ax4, fmt="d", cbar=False)
    ax4.set_title("Goal Difference Heatmap (Top 10 Teams)")
    plt.tight_layout()
    st.pyplot(fig4)

    st.markdown("### 5. Points Distribution")
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Pt."], bins=10, kde=True, ax=ax5, color="teal")
    ax5.set_title("Distribution of Points")
    ax5.set_xlabel("Points")
    plt.tight_layout()
    st.pyplot(fig5)

    st.markdown("### 6. Points vs Goal Difference (Top 10 Teams)")
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=top_teams, x="Dif", y="Pt.", hue="Team", s=150, ax=ax6)
    ax6.set_title("Points vs Goal Difference (Top 10 Teams)")
    ax6.set_xlabel("Goal Difference")
    ax6.set_ylabel("Points")
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig6)

# --------------------- FINALS HISTORY ---------------------
if analysis == "Finals History":
    st.subheader("Champions League Finals History")

    # Top 10 countries with most titles
    most_titles = df1['Country'].value_counts().head(10)
    st.write("🏆 Top 10 Countries with Most Titles")
    st.dataframe(most_titles)

    # Top 10 most frequent finalists
    team_counts = pd.concat([df1["Winners"], df1["Runners-up"]]).value_counts().head(10)
    st.write("🎯 Top 10 Most Frequent Finalists")
    st.dataframe(team_counts)

    # Finals decided in extra time
    extra_time_count = df1['Notes'].str.contains("extra time", case=False, na=False).sum()
    st.info(f"⏱ Finals decided in extra time: **{extra_time_count}**")

    # Most common final venue
    venue_counts = df1['Venue'].value_counts()
    st.write("🏟 Most Common Final Venue:")
    st.success(venue_counts.idxmax())

    st.markdown("### 1. Winner vs Runner-up Goal Distribution")
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df1[["Winner_Goals", "Runner_Goals"]], ax=ax7)
    ax7.set_title("Goal Distribution in Finals")
    ax7.set_xlabel("Team")
    ax7.set_ylabel("Goals")
    plt.tight_layout()
    st.pyplot(fig7) 

    st.markdown("### 2. Score Margin per Final (Last 20 Finals)")
    # Show only the last 20 finals for clarity
    last_20 = df1.tail(20)
    fig8, ax8 = plt.subplots(figsize=(12, 5))
    sns.barplot(data=last_20, x="Season", y="Score_Margin", palette="rocket", ax=ax8)
    ax8.set_title("Score Margins in Last 20 Finals")
    ax8.set_xlabel("Season")
    ax8.set_ylabel("Score Margin")
    ax8.set_xticklabels(last_20["Season"], rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig8)

    st.markdown("### 3. Top 10 Venues by Number of Finals Hosted")
    top_venues = venue_counts.head(10)
    fig9, ax9 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_venues.values, y=top_venues.index, palette="mako", ax=ax9)
    ax9.set_title("Top 10 Venues by Finals Hosted")
    ax9.set_xlabel("Number of Finals")
    ax9.set_ylabel("Venue")
    plt.tight_layout()
    st.pyplot(fig9) 
if analysis == "ML Prediction":
    st.subheader("🤖 Predict Winner: Team 1 vs Team 2 (Head-to-Head)")

    # Get unique team names
    team_names = sorted(df["Team"].dropna().unique())

    # User selects two teams
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1", team_names, key="team1")
    with col2:
        team2 = st.selectbox("Select Team 2", team_names, key="team2")

    # Prevent same team selection
    if team1 == team2:
        st.warning("Please select two different teams.")
    else:
        # Features to use
        perf_features = ["M.", "W", "D", "L", "goal", "Dif", "Pt.", "Goals_per_Match", "Win_Loss_Ratio"]

        # Prepare dataset for pairwise comparison
        df_pairs = []
        for i in range(len(df)):
            for j in range(len(df)):
                if i != j:
                    row1 = df.iloc[i]
                    row2 = df.iloc[j]
                    features = list(row1[perf_features]) + list(row2[perf_features])
                    label = 1 if row1["Pt."] > row2["Pt."] else 0  # 1 if team1 wins, else 0
                    df_pairs.append(features + [label])
        columns = [f"{f}_1" for f in perf_features] + [f"{f}_2" for f in perf_features] + ["Winner"]
        df_ml = pd.DataFrame(df_pairs, columns=columns)

        # Model training
        X = df_ml.drop("Winner", axis=1)
        y = df_ml["Winner"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        st.write(f"Model Test Accuracy: **{acc:.2%}**")

        # Prepare input for prediction
        row1 = df[df["Team"] == team1].iloc[0]
        row2 = df[df["Team"] == team2].iloc[0]
        input_features = list(row1[perf_features]) + list(row2[perf_features])
        input_scaled = scaler.transform([input_features])
        pred = clf.predict(input_scaled)[0]

        winner = team1 if pred == 1 else team2
        st.success(f"🏆 **Predicted Winner:** {winner}")