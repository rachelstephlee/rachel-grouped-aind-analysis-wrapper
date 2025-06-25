# aind-analysis-wrapper

This capsule is the 2nd one in this [pipeline](https://codeocean.allenneuraldynamics.org/capsule/8624294/tree) that runs the analysis on the input model and analysis specification.

### Setup
There are several things that need to be setup. In the environment tab, scroll to the section `Environment Variables` and set the `DOCDB_COLLECTION` variable to the project collection you want to write to. Be sure to also set the `CODEOCEAN_EMAIL` and the `ANALYSIS_BUCKET` variables to the desired values. In addition, you will need to create a secret for a code ocean token. See [CodeOcean docs](https://docs.codeocean.com/user-guide/code-ocean-api/authentication). After this, add any necessary packages you need to run analysis in the environment. 

### Running Analysis
In the `run_analysis` function in `analysis_wrapper/run_capsule.py`, add in relevant code and save necessary output to the `/results` folder. The results folder will be put in the output bucket specified in the analysis specification and the metadata for the analysis will be written to the docdb collection specified from the steps above.

### Inputs
There are 2 inputs currently to this capsule. The first is the output of the the [job dispatch capsule](https://codeocean.allenneuraldynamics.org/capsule/9303168/tree). The second is the analysis specification. Navigate to the `analysis_wrapper/example_analysis_model.py` and create your own analysis schema that will be used for seeing which analysis was run on which input. To do this, first define the schema in that file. Then, create a json and upload it to the data folder, naming it `analysis_parameters.json`. An example of a user defined json file for a given schema is below:

```json
{
    "analysis_name": "Unit Filtering",
    "analysis_tag": "Arjun's Filtering",
    "isi_violations_cutoff": 0.05
}
```
