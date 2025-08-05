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
The steps below are needed to configure the analysis wrapper. Go to the `environment` tab in the capsule. Steps 1-3 are found under **`Environment Variables`**. Step 4 is found under **`Secrets`**.
1. Set the **`DOCDB_COLLECTION`** to the project collection
2. Set the **`CODEOCEAN_EMAIL`** 
3. Replace the **`ANALYSIS_BUCKET`** with the path to the analysis bucket on S3.
4. Set the corresponding secrets. Users will have to make a codeocean token - **read** permissions are sufficient for this. See [creating token](https://docs.codeocean.com/user-guide/code-ocean-api/authentication#to-create-an-access-token) docs for more information.
5. Add any required packages needed for analysis

### Analysis Wrapper - User Defined Analysis Parameters
To help facilitate tracking of analysis parameters for reproducibility, a user should define their own pydantic model in the analysis wrapper. Follow steps below. An example can be found in `/code/example_analysis_model.py`:

1. Start by renaming the example analysis model to match user's custom model.
2. Then add any fields that would be useful in aiding reproducibility of analysis. **The listed fields are just examples, not a requirement**. ***Recommended to add a field to tag the version of analysis run***. See [Analysis Metadata Tracking](#analysis-metadata-tracking).
3. Additionally, for any small numerical outputs and such - define these in the output model if needed. Larger files such as arrays and tables should be written to the results folder (See next section).
4. Once this is done, be sure to replace **all references to `ExampleAnalysisSpecification` and `ExampleAnalysisOutputs`** in `run_capsule.py`. If no output model is needed, remove lines **38-43** in `run_capsule.py`.

### Running Analysis and Storing Output
User defined analysis can be specified in the **`run_analysis`** function in `run_capsule.py`. An example of the input model passed in can be found in `/data/job_dict`. An example of analysis parameters that correspond to the example pydantic model can be found in `/data/analysis_parameters.json`. Modify the analysis parameters json for testing if needed, and **make sure the fields match those in the model defined**. Some other notes below:

* **Users can also add an app panel for input arguments that are part of the analysis model**.

* **If there was no file extension specified when dispatching, change example in line 30 to s3_location. Then users will need to read from the S3 bucket directly**.

* **Results should be written to **`/results/`** folder in the capsule**. The results folder will then be copied to the S3 Analysis Bucket path set in the environment variables. This path will then be stored as part of the metadata record that will get written to the document database and can be queried later on.

* The metadata record is a combination of input data, analysis parameters, git commits, etc. All of these are used to query if analysis has already been run on the combination of input data, parameters, etc. ***IMPORTANT***: **BE SURE TO COMMIT ALL CHANGES IN THIS CAPSULE. IF CHANGES ARE NOT COMMITED AND ANALYSIS NEEDS TO BE RUN, IT COULD BE SKIPPED IF THE METADATA RECORD ALREADY EXISTS FOR THE GIVEN COMBINATION OF INPUT DATA, ANALYSIS PARAMETERS, CODE, ETC**

### Analysis Metadata Tracking

This framework automatically tracks completed analyses using a metadata record that includes:
- The input data
- The analysis parameters
- The committed code version
- The output S3 bucket containing analysis results

If all of these are the same, the system will **skip re-running** the analysis to avoid duplicate processing.

If you modify your analysis parameters or code, the system will consider it a **different record** and run it normally **after the changes in the capsule have been committed**.

To intentionally rerun the **same** analysis on the same data:
- Ensure that **at least one part of the metadata record changes** — this could be a version label, or a code comment.
- **Not required** but adding a versioning field in the user defined pydantic model is **strongly recommended** for reproducibility and distinguishing between analysis runs.

### Testing Analysis Wrapper
To test, a reproducible run can be executed. **When ready to run analysis and post results, be sure to set the dry run flag in the app panel to 0 so the results are posted. By default, dry run is enabled.**.
