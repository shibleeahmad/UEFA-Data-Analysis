# UEFA Champions League EDA & ML Prediction App

This Streamlit web app provides interactive **Exploratory Data Analysis (EDA)** and **Machine Learning (ML) predictions** for UEFA Champions League teams and finals, using historical data. 
live demo: https://uefa-data-analysis.streamlit.app/

## Features

- **Team Performance:** Visualize top teams' wins, losses, points, goal differences, and more.
- **Finals History:** Explore finals by country, venue, score margins, and other statistics.
- **Head-to-Head ML Prediction:** Select any two teams and predict the winner using their historical performance.

## How to Run

1. **Clone or Download** this repository.
2. Place `performance.csv` and `matches.csv` in the `IDS-project` folder.
3. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the app:
    ```bash
    streamlit run app.py
    ```

## File Structure

```
IDS-project/
├── app.py
├── performance.csv
├── matches.csv
├── requirements.txt
```

## Data Sources

- `performance.csv`: Team-level performance statistics.
- `matches.csv`: Historical Champions League finals data.

## Screenshots

*Add screenshots here if desired.*

## License

For educational and non-commercial use.

---

**Enjoy exploring and predicting the
