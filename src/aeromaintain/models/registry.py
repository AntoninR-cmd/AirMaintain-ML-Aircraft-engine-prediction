from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def create_models():
    return {
        "LinearRegression": LinearRegression(),

        "Ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0)
        ),

        "Lasso": make_pipeline(
            StandardScaler(),
            Lasso(
                alpha=0.01,
                max_iter=10_000,
                selection="random",
                random_state=42
            )
        ),

        "DecisionTree": DecisionTreeRegressor(
            max_depth=15,
            min_samples_leaf=5,
            random_state=42
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            early_stopping=True,
            random_state=42
        ),

        "SVR": make_pipeline(
            StandardScaler(),
            SVR(
                kernel="rbf",
                C=10,
                epsilon=5,
                gamma="scale",
                cache_size=4000
            )
        ),

        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(
                n_neighbors=10,
                weights="distance",
                n_jobs=-1
            )
        ),

        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=42
            )
        )
    }
