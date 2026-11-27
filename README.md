# TMDB Movie Data Analysis and Machine Learning

Programming for AI semester project. Instructor: Mahhek Tahir.

Team:Eisha Iqbal (25L-3003), Ifrah Noor (25L-0025)

This project collects data on movies released between 2016 and 2026 from The Movie Database (TMDB) API, cleans and explores it, and builds regression models that predict a movie's worldwide box office revenue using only information known before release.


| `notebooks/TMBD_Project.ipynb` | The full project notebook: data collection, understanding, cleaning, EDA, skewness, outliers, feature engineering and selection, train-test split, models, evaluation, advanced techniques and the two proof-of-work logs |
| `data/raw/tmdb_raw.csv` | Original raw dataset, one movie per row (1100 movies, 28 columns) |
| `data/raw/raw_json.zip` | Unmodified API responses, one JSON file per movie |
| `data/raw/movie_ids.csv` | The 1100 collected movie IDs with year and result page |
| `data/raw/discover_summary.csv` | Pages used and pages available per year |
| `data/raw/data_collection_log.txt`, `collection_log.json` | Data collection log |
| `data/processed/tmdb_zeros_as_missing.csv` | Dataset with TMDB zeros treated as missing and movies without revenue removed (899 movies) |
| `data/processed/tmdb_working_5pct_missing.csv` | Separate working copy with about 5% artificially introduced missing values |
| `data/processed/tmdb_cleaned_preprocessed.csv` | Cleaned, encoded and scaled dataset from the cleaning section |
| `data/processed/tmdb_invalid_removed.csv`, `tmdb_after_outlier_treatment.csv` | Datasets after the outlier analysis |
| `data/processed/tmdb_features_final.csv` | Final feature table used for modeling (876 movies) |
| `data/processed/train_preprocessed.csv`, `test_preprocessed.csv` | Training and test sets after preprocessing fitted on the training data |
| `reports/model_results.csv`, `model_training_log.txt` | Final test metrics of every model and the model training log |

## Data collection approach

- **Source:** TMDB API (`https://api.themoviedb.org/3`), endpoints `/discover/movie` and `/movie/{id}?append_to_response=credits`.
- **Date filters:** one calendar year at a time, from `primary_release_date.gte = YYYY-01-01` to `primary_release_date.lte = YYYY-12-31`, for 2016 to 2026.
- **Sorting:** `vote_count.desc`, with at least 10 votes and adult content excluded. Movies with more votes usually have more complete records, including budget and revenue, and vote count is more stable over time than popularity.
- **Pagination:** 20 results per page; 5 pages per year were needed to reach 100 unique movies.
- **Result:** 1100 unique movie IDs (100 per year, including the partial year 2026), 1210 requests in total, collected on 21 November 2026.
- **Details:** each movie's details and credits were requested in a single call and saved as raw JSON with a `collected_at` timestamp.
- **Fingerprint:** SHA-256 of `tmdb_raw.csv`: `45903e104914bc20dc3352cf6914673c628396a9ecb1d7fd110cd92a167b5447`

The collection code is in the notebook (data collection section), together with its outputs and the data collection log.

## Method summary

1. **Cleaning:** TMDB zeros in budget and revenue treated as missing; movies without revenue removed; 5% artificial missing values imputed in a separate working copy; one-hot and multi-hot encoding; scaling.
2. **Exploration:** histograms, box plots, scatter plots, bar charts and a correlation heatmap; skewness analysis and transformations; outliers investigated with the IQR method before treatment.
3. **Scope:** one movie with impossible values (budget and revenue of 7 dollars) and 22 streaming releases with token cinema revenues were removed, leaving 876 movies.
4. **Features:** date parts, release season flags, counts of genres, companies, countries and languages, collection and language flags, genre columns and a missing-budget flag. Audience statistics (votes, popularity) were excluded because they are only known after release.
5. **Selection:** forward selection and backward elimination on the training data; the 10 features chosen by both methods were kept.
6. **Split:** 80/20 train-test split stratified on revenue groups (700 / 176 movies); all preprocessing fitted on the training data only.
7. **Models:** ridge regression, random forest and gradient boosting, evaluated with MAE, MSE, RMSE, R², cross-validation, hyperparameter tuning and residual analysis.
8. **Advanced techniques:** Yeo-Johnson power transform and missingness pattern analysis.

## Results

| Model | Test MAE (M$) | Test RMSE (M$) | Test R² |
|---|---|---|---|
| Ridge (sqrt/log features, log target) | 128.6 | 246.2 | 0.554 |
| Random Forest | 141.5 | 272.2 | 0.455 |
| Gradient Boosting | 133.8 | 256.8 | 0.515 |
| Tuned Gradient Boosting | 135.5 | 270.4 | 0.462 |
| Ridge + Yeo-Johnson features (log target) | 130.6 | 252.3 | 0.532 |
| Tuned Gradient Boosting + Yeo-Johnson | 134.7 | 263.9 | 0.488 |
| **Final: Ridge + Yeo-Johnson features and target** | **134.0** | **261.1** | **0.498** |

The final model has the lowest cross-validated MAE (116.9 million dollars) and the most stable cross-validated R² (0.51 ± 0.09) of all models. The Yeo-Johnson transform removed the unrealistic predictions that plain ridge regression made for films with exceptionally large budgets. Budget and collection membership are the strongest drivers of revenue.

## Reproducibility

- Random seed: `0025` (last four digits of the lowest team roll number), used for every random state.
- The notebook runs in Google Colab with Google Drive mounted. The TMDB token is stored in Colab Secrets as `TMDB_TOKEN` and never appears in the code.
- Recovery cells near the top of the notebook reload the saved datasets after a Colab restart, so later sections can be rerun without collecting the data again.

## Limitations

The sample contains the most voted films of each year, so it leans toward well known titles. TMDB budgets and revenues are user contributed and sometimes missing or wrong, and revenues are not adjusted for inflation. The model only uses pre-release information, so it cannot anticipate breakout hits or expensive flops.
