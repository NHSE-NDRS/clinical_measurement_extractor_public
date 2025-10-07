import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.metrics import precision_recall_fscore_support

from src.load_data import load_dataframe_from_s3
from src.evaluation_helpers import add_complete_match

class CMEEvaluator:
    """
    This Class will take in a dataframe created from the clinical measurement extractor.

    You will need to pass in a comparison dictionary which will provide the class the information on
    which columns are being compared.
    """
    def __init__(
        self,
        comparison_dict: dict,
        accepted_values: dict,
        id_col: str,
        df: pd.DataFrame = None,
        bucket_name: str = None,
        folder: str = None,
        object_key: str = None,
        list_saved: bool = False
    ):
        """
        Initialize the EvaluatorClass using comparison dict and from:
        - df
        - bucket_name + object_key
        - list_saved = True + bucket_name (prompts user selection)

        Args:
            comparison_dict (dict): Column mapping for comparison.
            accepted_values (dict): Dictionary of a list of accepted values for each key in the JSON.
            id_col (str): Name of the ID column in the dataframe.
            df (pd.DataFrame): Raw input dataframe.
            bucket_name (str): S3 bucket name.
            folder (str): Name of your S3 folder
            object_key (str): S3 key to the summary_df file.
            list_saved (bool): If True, will list S3 summary files to choose from.
        """
        self.comparison_dict = comparison_dict
        self.accepted_values = accepted_values
        
        if (
            df is not None and 
            not list_saved and 
            bucket_name is None and 
            folder is None and
            object_key is None
        ):
            pass
        
        elif (
            list_saved and 
            bucket_name is not None and 
            folder is not None and
            df is None and 
            object_key is None
        ):
            df = self._load_from_s3_prompt(bucket_name, folder)
        
        elif (
            bucket_name is not None and 
            object_key is not None and 
            folder is not None and
            not list_saved and 
            df is None
        ):
            folder_object_key = f"{folder}/{object_key}"
            df = load_dataframe_from_s3(bucket_name, folder_object_key)
        else:
            raise ValueError("Must provide either (df), (bucket_name and object_key), or set (list_saved=True and bucket_name.)")
            
        self._init_from_df(df, id_col)  
        sns.set_style("whitegrid")

    def _init_from_df(self, df: pd.DataFrame, id_col: str):
        """
        Initialises the data from loading the evaluation dataframe locally.
        """
        self.df = self._add_comparison_columns(df.copy())
        self.id_col = id_col
        self.match_cols = [f"{orig}_V_{comp}" for orig, comp in self.comparison_dict.items()]
        self.text_display_cols = [self.id_col, "model_output", "parsed_output", "validated_output"]

        # Calculates the number of correct values for each entitiy and each row.
        self.rowwise_distribution = self._compute_rowwise_distribution(self.df)
        # This calculates the number of correct values for each column for each entity.
        self.correctness_summary_df = self._compute_correctness_summary_df()
        # This calculates the count of assigned values for each entity for each row.
        self.confusion_matrices = {
            col: self._compute_confusion_matrix_data(col)
            for col in self.comparison_dict.keys()
        }
        # This calculates the precision, recall, and f1 for each correctly assigned V incorrectly assigned value.
        self.per_value_metrics = {
            col: self._compute_per_value_metrics(col)
            for col in self.comparison_dict.keys()
        }

    def _load_from_s3_prompt(self, bucket_name: str, folder: str):
        """
        Lets the user choose a CSV object from a specified folder in an S3 bucket to load.
        """
        s3 = boto3.client('s3')
    
        prefix = folder
        if not prefix.endswith('/'):
            prefix += '/'
    
        # List objects with the folder prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        contents = response.get('Contents', [])
        
        # Filter keys that end with .csv and are inside the folder (prefix)
        csv_keys = [obj['Key'] for obj in contents if obj['Key'].endswith('.csv')]
    
        if not csv_keys:
            raise ValueError(f"No CSV files found in s3://{bucket_name}/{prefix}")
    
        # Get descriptions from metadata for each CSV
        descriptions = []
        for key in csv_keys:
            obj = s3.head_object(Bucket=bucket_name, Key=key)
            meta = obj.get("Metadata", {})
            desc = meta.get("description", "(no description)")
            descriptions.append((key, desc))
    
        # Show choices to user
        print("\nAvailable CSV files in folder:\n")
        for i, (key, desc) in enumerate(descriptions):
            print(f"{i+1}. {key} — {desc}")
    
        choice = input("\nEnter the number of the CSV you want to load: ")
        try:
            index = int(choice) - 1
            selected_key = descriptions[index][0]
        except (ValueError, IndexError):
            raise ValueError("Invalid selection.")
        dict_cols = list(self.comparison_dict) + list(self.comparison_dict.values())
        df = load_dataframe_from_s3(bucket_name, selected_key, str_columns = dict_cols)
        return df

    def _add_comparison_columns(self, df:pd.DataFrame):
        """
        This does an exact comparison match, and also accounts for Nans.
        """
        df = df.copy()
        for original_col, compare_col in self.comparison_dict.items():
            match_col = f"{original_col}_V_{compare_col}"
            df = add_complete_match(df, original_col, compare_col, match_col)
        return df

    def _compute_rowwise_distribution(self, df: pd.DataFrame):
        """
        Calculates how many of the comparisons are correct per row,
        and returns a complete distribution (including missing counts as 0),
        along with human-readable labels.
        """
        df = df.copy()
        df["_num_correct"] = df[self.match_cols].sum(axis=1)
    
        num_comparisons = len(self.comparison_dict)
        full_range = list(range(num_comparisons + 1))
    
        # Actual counts
        summary = (
            df.groupby("_num_correct")
            .size()
            .reindex(full_range, fill_value=0)
            .reset_index(name="count")
        )
    
        summary["percent"] = ((summary["count"] / summary["count"].sum()) * 100).round(2)
    
        def format_label(num_correct):
            if num_correct == num_comparisons:
                return "All correct"
            elif num_correct == 0:
                return "None correct"
            else:
                return f"{num_correct} correct"
    
        summary["label"] = summary["_num_correct"].apply(format_label)
    
        return summary.to_dict(orient="records")

    def _compute_correctness_summary_df(self):
        """
        Computes percentage correctness for each original vs compare column pair.
        Returns a DataFrame with correct percentages and metadata.
        """
        data = []
        for i, (orig_col, comp_col) in enumerate(self.comparison_dict.items(), start=1):
            match_col = f"{orig_col}_V_{comp_col}"
            col_df = self.df.copy()
            total = col_df[match_col].count() # Get's total count.
            correct_total = col_df[match_col].sum() # Correct is true, which is 1.
            correct_percent = ((correct_total / total)* 100).round(2) if total else 0.0
    
            data.append({
                "comparison_index": i,
                "original_col": orig_col,
                "compare_col": comp_col,
                "match_col": match_col,
                "correct_percent": correct_percent
            })
    
        return pd.DataFrame(data)
        
    def _compute_confusion_matrix_data(self, original_col: str):
        """
        For a given metric, it compares the actual and extracted values. 
        This generates a union distinct across the actual and extracted values, to generate a matrix.
    
        Additional handling:
        - If 'status' == 'validation_failed', y_pred is set to 'validation_failed' before NaN fill.
        - If y_pred is not in accepted values, set to 'non-accepted values'.
        """
    
        compare_col = self.comparison_dict[original_col]
    
        # Strictly only the real accepted values (no validation_failed)
        accepted_values = [str(val) for val in self.accepted_values.get(compare_col, [])]
    
        df = self.df.copy()
    
        # Work with the raw series first (don't convert to str yet)
        y_true = df[original_col]
        y_pred = df[compare_col]
    
        # Apply validation_failed override BEFORE missing value handling (use mask to avoid SettingWithCopy)
        y_pred = y_pred.mask(df["status"] == "validation_failed", "validation_failed")
    
        # Normalize missing values (this will catch np.nan, pd.NA and Python None)
        y_true = y_true.fillna("true_na")
        y_pred = y_pred.fillna("true_na")
    
        # Now convert to str (safe) and also catch any literal "nan" / "None" strings
        y_true = y_true.astype(str).replace({"nan": "true_na", "None": "true_na"})
        y_pred = y_pred.astype(str).replace({"nan": "true_na", "None": "true_na"})
    
        # After handling NA and overrides, apply acceptance filter
        y_pred = y_pred.where(
            y_pred.isin(accepted_values + ["true_na", "key_missing", "validation_failed"]),
            "non_accepted"
        )
        
        # Final label definitions
        y_labels = accepted_values  + ["true_na"]
        special_labels = ["key_missing", "validation_failed", "non_accepted"]
        x_labels = y_labels + special_labels
    
        # Build confusion matrix
        cm_df = (
            pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
            .groupby(['y_true', 'y_pred'])
            .size()
            .unstack(fill_value=0)
            .astype(int)
        )
    
        # Ensure labels are unique and in order
        cm_df = cm_df.reindex(index=y_labels, columns=x_labels, fill_value=0)
    
        return cm_df

    def _compute_per_value_metrics(self, original_col:str):
        """
        This calculates the precision, recall, and f1_score for each
        distinct value per comparison.
        """
        compare_col = self.comparison_dict[original_col]
    
        # Replace NaN with string "nan" to treat as a valid category
        y_true = self.df[original_col].fillna("true_na").astype(str)
        y_pred = self.df[compare_col].fillna("true_na").astype(str)
    
        # Unique labels from both y_true and y_pred
        labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
    
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
    
        results = []
        for label in labels:
            idx = labels.index(label)
            results.append({
                "value": label,
                "precision": (precision[idx]*100).round(2),
                "recall": (recall[idx]*100).round(2),
                "f1_score": (f1[idx]*100).round(2),
                "support": int(support[idx])
            })
    
        return results

    def get_status_summary(self, stat_col: str):
        """ Get the Summary of Statuses across the run """
        df = self.df.copy()
        return pd.DataFrame(df[stat_col].value_counts())

    def get_validation_failed(self, stat_col: str):
        """ Display all that failed validation """
        df = self.df.copy()
        sub_df = df[self.text_display_cols + [stat_col]]
        return sub_df[sub_df[stat_col] == "validation_failed"]

    def get_invalid(self, stat_col: str):
        """ Display all invalid JSONs"""
        df = self.df.copy()
        sub_df = df[self.text_display_cols + [stat_col]]
        return sub_df[sub_df[stat_col] == "invalid"]

    def get_partial(self, stat_col: str):
        """ Display all partially structured JSONs"""
        df = self.df.copy()
        sub_df = df[self.text_display_cols + [stat_col]]
        return sub_df[sub_df[stat_col] == "partial"]

    def print_text(self, id_val:str, text_col:str):
        """ Display specific text to user by column id """
        df = self.df.copy()
        print(df[df[self.id_col].astype(str) == f"{id_val}"].reset_index().iloc[0][text_col])

    def get_non_accepted_summary(self, actual_col: str, sort_by_count=True):
        """
        Creates a summary dataframe of actual vs. non-accepted predicted values,
        excluding 'key_missing', 'validation_failed', and 'true_na' from the summary,
        sorted by count descending.
        """
        compare_col = self.comparison_dict[actual_col]
        accepted_values = set(str(v) for v in self.accepted_values.get(compare_col, []))
    
        special_exclude = ["key_missing", "validation_failed", "true_na"]
    
        df = self.df.copy()
    
        y_true = df[actual_col].fillna("true_na").astype(str)
        y_pred = df[compare_col].fillna("true_na").astype(str)
    
        # Filter predicted values that are NOT accepted AND NOT in special_exclude
        mask = (~y_pred.isin(accepted_values)) & (~y_pred.isin(special_exclude))
    
        filtered_df = pd.DataFrame({
            "text_id": df[self.id_col][mask],
            "actual_value": y_true[mask],
            "extracted_value": y_pred[mask]
        })
    
        summary_df = (
            filtered_df
            .groupby(["actual_value", "extracted_value"])
            .agg(
                count=("text_id", "size"),
                text_ids=("text_id", list)
            )
            .reset_index()
        )
    
        if sort_by_count:
            summary_df = summary_df.sort_values(by="count", ascending=False)
    
        return summary_df

    def get_non_accepted_summary_all(self, sort_by_count: bool = True, group_sort: bool = True) -> pd.DataFrame:
        """
        Runs get_non_accepted_summary for each actual_col in actual_cols,
        adds a column to identify actual_col, and concatenates all summaries.
    
        Args:
            sort_by_count (bool): Whether to sort by 'count' descending, by default True.
            group_sort (bool): Whether to sort by ['actual_col', 'actual_value'] grouping first, by default True.
    
        Returns:
            pd.DataFrame: Concatenated summary of non-accepted values with actual_col and text_id_list included.
        """
        dfs = []
        actual_cols = list(self.comparison_dict.keys())
        
        for actual_col in actual_cols:
            summary_df = self.get_non_accepted_summary(actual_col, sort_by_count=False)
            summary_df["actual_col"] = actual_col
            dfs.append(summary_df)
    
        combined_df = pd.concat(dfs, ignore_index=True)
    
        if group_sort:
            sort_cols = ["actual_col", "actual_value"]
            if sort_by_count:
                sort_cols.append("count")
                combined_df = combined_df.sort_values(sort_cols, ascending=[True, True, False])
            else:
                combined_df = combined_df.sort_values(sort_cols)
        else:
            if sort_by_count:
                combined_df = combined_df.sort_values("count", ascending=False)
    
        combined_df = combined_df.reset_index(drop=True)
    
        # Return relevant columns, including text_id_list
        return combined_df[[
            "actual_col", "actual_value", "extracted_value", "count", "text_ids"
        ]]

    def plot_correctness_and_rowwise_distribution(self):
        """
        Plots row-wise distribution (how many comparisons are correct per row)
        and correctness per comparison column.
        """

        # load data
        rowwise_df = pd.DataFrame(self.rowwise_distribution)
        correct_df = self.correctness_summary_df
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 2]})
    
        # Plot 1: Row-wise Distribution
        sns.barplot(x="label", y="percent", hue="label", data=rowwise_df, palette="Blues", ax=ax1)
        ax1.set_title("Percentage of Correct Metric Count across Reports", fontsize=14)
        ax1.set_ylabel("Percentage of Reports")
        ax1.set_xlabel("Correct Metric Count per Report")
        ax1.set_ylim(0, 100)
        ax1.get_legend()
    
        for bar, pct in zip(ax1.patches, rowwise_df["percent"]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{pct:.1f}%", ha="center", fontsize=9)
    
        # Plot 2: Correctness per Comparison Column
        bars = ax2.bar(correct_df["comparison_index"], correct_df["correct_percent"], color="#5B9BD5")
        ax2.set_title("Percentage Correct per Comparison Column", fontsize=14)
        ax2.set_ylabel("Correct (%)")
        ax2.set_xlabel("Comparison Index")
        ax2.set_ylim(0, 100)
        ax2.set_xticks(correct_df["comparison_index"])
        ax2.set_xticklabels(correct_df["comparison_index"])
    
        for bar, val in zip(bars, correct_df["correct_percent"]):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}%", ha="center", fontsize=9)
    
        legend_labels = [
            f"{i+1}. {row['original_col']} vs {row['compare_col']}"
            for i, row in correct_df.iterrows()
        ]
        legend_patches = [
            mpatches.Patch(color="#5B9BD5", label=label)
            for label in legend_labels
        ]
        ax2.legend(
            handles=legend_patches,
            title="Comparison Index Mapping",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=2,
            frameon=True
        )
    
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()

    def plot_per_metric_plots(self, original_col:str):
        """
        Plots the confusion matrix and precision/recall/f1 for each distinct value.
        """
        cm_df = self.confusion_matrices[original_col]
        compare_col = self.comparison_dict[original_col]
        metrics_data = self.per_value_metrics[original_col]
    
        accepted_values = [str(v) for v in self.accepted_values.get(compare_col, [])] 
        
        # Fixed label order
        special_labels = ["true_na", "key_missing", "validation_failed", "non_accepted"]
        labels_ordered = accepted_values + special_labels

        cm_df = cm_df.reindex(index=accepted_values + ["true_na"], columns=labels_ordered, fill_value=0)

        metrics_df = pd.DataFrame(metrics_data).set_index("value").reindex(accepted_values + ["true_na"]).dropna().reset_index()

        color_map = {
            "precision": "#A8D5BA",
            "recall": "#FFF4B2",
            "f1_score": "#F4A582"
        }
    
        metrics = color_map.keys()
        num_groups = len(metrics_df["value"])
        num_metrics = len(metrics)
        width = 0.2
        gap = 0.3
        x = np.arange(num_groups) * (num_metrics * width + gap)
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 1.6]})

        sns.heatmap(
            cm_df.astype(int),
            annot=cm_df.astype(int),
            fmt="d",
            cmap="YlGnBu",
            cbar=True,
            linewidths=0.5,
            linecolor='gray',
            square=True,
            ax=ax1
        )
        ax1.set_title(f"Actual V Extracted Matrix\n'{original_col}' vs. '{compare_col}'", fontsize=12)
        ax1.set_xlabel(f"Extracted ({compare_col})")
        ax1.set_ylabel(f"Actual ({original_col})")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
        # Per-Value Metrics
        for i, metric in enumerate(metrics):
            ax2.bar(x + i * width, metrics_df[metric], width, label=metric.title(), color=color_map.get(metric, "gray"))
    
        ax2.set_xticks(x + width * (num_metrics - 1) / 2)
        ax2.set_xticklabels(metrics_df["value"], rotation=0, ha="right")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Percentage (%)")
        ax2.set_title(f"Per-Value Metrics for '{original_col}'", fontsize=12)
        ax2.legend(title="Metric", loc="upper right")
    
        plt.tight_layout()
        plt.show()


    def plot_per_metric_plots_for_all(self):
        """
        Plots per-metric analysis for all comparisons.
        """
        for original_col in self.comparison_dict:
            self.plot_per_metric_plots(original_col=original_col)