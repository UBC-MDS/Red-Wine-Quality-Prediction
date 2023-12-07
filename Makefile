# Makefile for the Red Wine Quality Prediction to execute report using Jupyter Book, or clean (remove) files.

# Make all
all: report/_build/html/index.html

# Repeating histograms for each variable in the dataset
results/figures/repeating_hists_plot.png: scripts/plot_repeating_hists.py data/winequality-red.csv
	python scripts/plot_repeating_hists.py \
		data/winequality-red.csv \
		results/figures/repeating_hists_plot.png

# Split data into train and test sets
results/tables/X_train.csv results/tables/X_test.csv results/tables/y_train.csv results/tables/y_test.csv: scripts/data_split.py data/winequality-red.csv
	python scripts/data_split.py \
		data/winequality-red.csv \
		X_train.csv \
		X_test.csv \
		y_train.csv \
		y_test.csv \
		0.3 \
		522s

# Set dummy classifier as baseline model and return cross-validate results
results/tables/cv_results.csv: scripts/baseline_model.py results/tables/X_train.csv results/tables/y_train.csv
	python scripts/baseline_model.py \
		X_train.csv \
		y_train.csv \
		cv_results.csv

# Model tuning for logistic regression model, decision tree model, KNN model and SVC model.
# Each generates a table for the details of the best performing one.
results/tables/logistic_grid_search.csv: scripts/model_hyperparam_tuning_wrapper.py results/tables/X_train.csv results/tables/y_train.csv
	python scripts/model_hyperparam_tuning_wrapper.py \
		results/tables/X_train.csv \
		results/tables/y_train.csv \
		logistic \
		results/tables/

results/tables/decision_tree_grid_search.csv: scripts/model_hyperparam_tuning_wrapper.py results/tables/X_train.csv results/tables/y_train.csv
	python scripts/model_hyperparam_tuning_wrapper.py \
		results/tables/X_train.csv \
		results/tables/y_train.csv \
		decision_tree \
		results/tables/

results/tables/knn_grid_search.csv: scripts/model_hyperparam_tuning_wrapper.py results/tables/X_train.csv results/tables/y_train.csv
	python scripts/model_hyperparam_tuning_wrapper.py \
		results/tables/X_train.csv \
		results/tables/y_train.csv \
		knn \
		results/tables/

results/tables/svc_grid_search.csv: scripts/model_hyperparam_tuning_wrapper.py results/tables/X_train.csv results/tables/y_train.csv
	python scripts/model_hyperparam_tuning_wrapper.py \
		results/tables/X_train.csv \
		results/tables/y_train.csv \
		svc \
		results/tables/

# Combine the four model tunning results into one table
results/tables/comparison_df.csv: scripts/model_table_combination.py results/tables/
	python scripts/model_table_combination.py \
		results/tables/ \
		results/tables/

# Choose the best performing model: SVC and fit on test data 
results/tables/test_set_score.csv: scripts/test_set_deployment.py results/tables/ results/tables/ results/tables/ results/tables/ results/tables/
	python scripts/test_set_deployment.py \
		results/tables/ \
		results/tables/ \
		results/tables/ \
		results/tables/ \
		results/tables/ \
		results/tables/

# Confusion matrix for the best model SVC performance on the test data
results/figures/confusion_matrix_plot.png: scripts/confusion_matrix.py results/models/best_pipe.pickle results/tables/X_test.csv results/tables/y_test.csv
	python scripts/confusion_matrix.py \
		--model=results/models/best_pipe.pickle \
		--x_test_path=results/tables/X_test.csv \
		--y_test_path=results/tables/y_test.csv \
		--output_file=results/figures/confusion_matrix_plot.png

# Correlation matrix for all red wine physiochemical features in the data frame
results/figures/correlation_matrix_plot.png: scripts/correlation_matrix.py data/winequality-red.csv
	python scripts/correlation_matrix.py \
		data/winequality-red.csv \
		results/figures/correlation_matrix_plot.png

# write the HTML report with Jupyter Book
report/_build/html/index.html: report/red_wine_quality_prediction.ipynb \
	report/_toc.yml \
	report/references.bib \
	report/_config.yml \
	data/winequality-red.csv \
	results/figures/repeating_hists_plot.png \
	results/tables/X_train.csv\
	results/tables/X_test.csv \
	results/tables/y_train.csv \
	results/tables/y_test.csv \
	results/tables/cv_results.csv \
	results/tables/logistic_grid_search.csv \
	results/tables/decision_tree_grid_search.csv \
	results/tables/knn_grid_search.csv \
	results/tables/svc_grid_search.csv \
	results/tables/comparison_df.csv \
	results/tables/test_set_score.csv \
	results/models/best_pipe.pickle \
	results/figures/confusion_matrix_plot.png \
	results/figures/correlation_matrix_plot.png
		jupyter-book build report
		cp -r report/_build/html/* docs
		if [ ! -f ".nojekyll" ]; then touch docs/.nojekyll; fi

# Delete all existing files (listed below) with Clean
clean:
	rm -f results/figures/repeating_hists_plot.png
	rm -f results/tables/X_train.csv
	rm -f results/tables/X_test.csv
	rm -f results/tables/y_train.csv
	rm -f results/tables/y_test.csv
	rm -f results/tables/cv_results.csv
	rm -f results/tables/logistic_grid_search.csv
	rm -f results/tables/decision_tree_grid_search.csv
	rm -f results/tables/knn_grid_search.csv
	rm -f results/tables/svc_grid_search.csv
	rm -f results/tables/comparison_df.csv
	rm -f results/tables/test_set_score.csv
	rm -f results/models/best_pipe.pickle
	rm -f results/figures/confusion_matrix_plot.png
	rm -f results/figures/correlation_matrix_plot.png
	rm -rf report/_build \
		docs/*

