import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self,):
        self.mappings_ = {}
        
    def fit(self, X, y=None):
        X = X.copy()

        # Identify and store mappings for categorical columns
        for c in X.columns:
            if X[c].dtype == 'object':
                X[c], uniques = X[c].factorize()
                X[c] = X[c].astype("int32")
                self.mappings_[c] = dict(zip(uniques, range(len(uniques))))
            else:
                X[c] = X[c].astype("int32")

        return self

    def _generate_string_based_features(self, X):
        """These rely on categories BEFORE factorization."""
        X = X.copy()

        X["is_low_visibility"] = X["lighting"].isin(["night", "dim"]).astype("int32")
        X["bad_weather"] = X["weather"].isin(["rainy", "foggy"]).astype("int32")
        X["rush_hour"] = X["time_of_day"].isin(["morning", "evening"]).astype("int32")

        return X

    def _generate_numeric_features(self, X):
        """These rely on numerical columns AFTER factorization."""
        X = X.copy()

        X["sharp_curve"] = (X["curvature"] > 0.6).astype("int32")
        X["high_speed"] = (X["speed_limit"] > 60).astype("int32")
        X["speed_curvature"] = X["speed_limit"] * X["curvature"]
        X["speed_dark"] = X["speed_limit"] * X["is_low_visibility"]
        X["curve_dark"] = ((X["curvature"] > 0.6) & (X["is_low_visibility"] == 1)).astype("int32")
        X["curvature_group"] = (X["curvature"] * 5).astype("int32").clip(0, 4)

        return X

    def transform(self, X):
        X = X.copy()

        # 1️⃣ Generate string-based features FIRST
        X = self._generate_string_based_features(X)

        # 2️⃣ Apply factorization mappings
        for c, mapping in self.mappings_.items():
            X[c] = X[c].map(mapping).fillna(-1).astype("int32")

        # 3️⃣ Generate numeric-only features
        X = self._generate_numeric_features(X)
          # Convert bool → int32
        bool_cols = X.select_dtypes(include=["bool"]).columns
        for bc in bool_cols:
            X[bc] = X[bc].astype("int32")

        for col in list(X.columns):
            if X[col].dtype == "object":
                X[col] = pd.Categorical(X[col]).codes.astype("int32")
        
        return X
        



class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, alpha=10):
        self.cols = cols
        self.alpha = alpha
        self.global_mean_ = None
        self.encodings_ = {}

    def fit(self, X, y):
        X = X.copy()
        
        # store global mean
        self.global_mean_ = y.mean()

        # attach target to X for groupby
        X["_target_"] = y

        for col in self.cols:
            
            #  Handle list of columns → combined feature
    
            if isinstance(col, list):
                combined_name = "__".join(col)
                X[combined_name] = (
                    X[col].astype(str).agg("__".join, axis=1)
                )
                group_col = combined_name
                key = tuple(col)
            else:
                group_col = col
                key = col

            
            #  Compute aggregated stats
            
            stats = (
                X.groupby(group_col)["_target_"]
                .agg(["mean", "count"])
                .reset_index()
            )

            # smoothed average
            stats["smooth"] = (
                (stats["count"] * stats["mean"] + self.alpha * self.global_mean_)
                / (stats["count"] + self.alpha)
            )

            # map category → smooth value
            self.encodings_[key] = dict(zip(stats[group_col], stats["smooth"]))

        return self
    




    def transform(self, X):
        X = X.copy()

        for col in self.cols:
            if isinstance(col, list):
                combined_name = "__".join(col)
                X[combined_name] = X[col].astype(str).agg("__".join, axis=1)
                key = tuple(col)
                new_col = "TE_" + combined_name
                source_col = combined_name
            else:
                key = col
                new_col = "TE_" + col
                source_col = col

            X[new_col] = (
                X[source_col].map(self.encodings_[key])
                .fillna(self.global_mean_).astype("float32")
            )
            if isinstance(col, list) and combined_name in X.columns:
                X = X.drop(columns=combined_name, axis=1)

        return X