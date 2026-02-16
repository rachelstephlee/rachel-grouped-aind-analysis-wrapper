# aind-analysis-wrapper

⚠️ **IMPORTANT: This is a Template Repository** ⚠️

This repository serves as an **example template** for building your own analysis workflows. **You should duplicate this repository and customize it for your specific analysis needs.** Do not modify this template directly - instead, create your own copy and build your analysis on top of the provided framework.

The **analysis wrapper** is a standardized framework for running large-scale data analysis workflows on cloud infrastructure. It processes job input models from the [job dispatcher](https://github.com/AllenNeuralDynamics/aind-analysis-job-dispatch), executes your custom analysis code, and automatically handles metadata tracking and result storage.

### What it does

The analysis wrapper:
1. **Receives** job input models containing data file locations and analysis parameters
2. **Executes** your custom analysis code on the specified datasets
3. **Tracks** metadata including inputs, parameters, code versions, and execution details
4. **Stores** results to cloud storage and writes metadata records to a document database
5. **Prevents** duplicate processing by checking if analysis has already been completed

### Environment Setup
The steps below are needed to configure the analysis wrapper. Go to the `/code/settings.env` file. Set the necessary fields in this file. In addition, be sure to set the secrets in the capsule settings.

### Analysis Wrapper - User Defined Analysis Parameters
To help facilitate tracking of analysis parameters for reproducibility, a user should define their own pydantic model in the analysis wrapper. Follow steps below. An example can be found in `/code/run_capsule.py`:

1. Start by renaming the example analysis parameters model to match user's custom model.
2. Additionally, for any small numerical outputs and such - define these in the output model if needed. Larger files such as arrays and tables should be written to the results folder (See next section).
3. Once this is done, be sure to replace **all references to `ExampleAnalysisParameters` and `ExampleAnalysisOutputs`** in `run_capsule.py`. If no output model is needed, remove lines referencing **ExampleAnalysisOutputs** in **`run_analysis`** function.

### Running Analysis and Storing Output
User defined analysis can be specified in the **`run_analysis`** function in `run_capsule.py`. An example of the input model passed in can be found in `/code/example_input/example_dispatch_job.json`.

* **Users can also add an app panel for input arguments that are part of the analysis model**.

* **If there was no file extension specified when dispatching, change to `analysis_dispatch_inputs.s3_location` in for loop in run_analysis function. Then users will need to read from the S3 bucket directly**.
* An example has been provided of fetching metadata records from the dispatch model. This will return a list of dictionaries where each item will contain the full metadata record. 

* **Run_analysis should return the output parameters AND results should be written to **`/results/`** folder in the capsule**. The results folder will then be copied to the S3 Analysis Bucket path set in the environment variables. This path will then be stored as part of the metadata record that will get written to the document database and can be queried later on.

* The metadata record is a combination of input data, analysis parameters, git commits, etc. All of these are used to query if analysis has already been run on the combination of input data, parameters, etc. ***IMPORTANT***: **BE SURE TO COMMIT ALL CHANGES IN THIS CAPSULE. IF CHANGES ARE NOT COMMITED AND ANALYSIS NEEDS TO BE RUN, IT COULD BE SKIPPED IF THE METADATA RECORD ALREADY EXISTS FOR THE GIVEN COMBINATION OF INPUT DATA, ANALYSIS PARAMETERS, CODE, ETC**

### Testing Analysis Wrapper
To test and run at the pipeline level, a reproducible run needs to be executed. **When ready to run analysis and post results, be sure to set the dry run flag in the app panel to 0 so the results are posted. By default, dry run is enabled.**.
