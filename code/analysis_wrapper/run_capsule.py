import logging
import os
from pathlib import Path


import sys
script_dir = "/root/capsule/code/analysis_wrapper"
if script_dir in sys.path:
    sys.path.remove(script_dir)
# sys.path.insert(0,"/root/capsule/")

sys.path.insert(0, str(Path(__file__).resolve().parent))


from analysis_pipeline_utils.analysis_dispatch_model import AnalysisDispatchModel
from analysis_pipeline_utils.utils_analysis_wrapper import run_analysis_jobs

from analysis_wrapper.analysis_model import (
    SummaryPlotsAnalysisSpecification, SummaryPlotsAnalysisOutputs
)

from rachel_analysis_utils import nwb_utils as r_utils
from rachel_analysis_utils import analysis_utils
from rachel_analysis_utils import data_curation_helpers
from analysis_wrapper.plots import summary_plots



DATA_PATH: Path = Path("/data")  # TODO: don't hardcode
ANALYSIS_BUCKET = os.getenv("ANALYSIS_BUCKET")
logger = logging.getLogger(__name__)

def validate_pearsonr(parameters):
    channel_keys = set(parameters['channels'].keys())
    pair_keys = set(dict.fromkeys(ch for pair in parameters['pearson_pairs'] for ch in pair))

    missing = pair_keys - channel_keys
    if missing:
        print("Missing channels referenced in parameters['pearson_pairs']:", missing)
        # drop any pairs that reference missing channels
        valid_pairs = [pair for pair in parameters.get('pearson_pairs', []) if all(ch in channel_keys for ch in pair)]
        removed = [pair for pair in parameters.get('pearson_pairs', []) if pair not in valid_pairs]
        if removed:
            print("Removing pairs with missing channels:", removed)
        parameters['pearson_pairs'] = valid_pairs

    return parameters

def get_all_channels(parameters, ch_suffix, df_fip=None):
    """
    Channels to plot, paired with their location labels.

    Under curation the names come from the curated df_fip events (targets), and
    the labels in parameters['channels'] no longer apply.
    """
    if df_fip is not None:
        all_channels = sorted(df_fip['event'].unique())
        return all_channels, ['' for _ in all_channels]

    all_channels = [ch + ch_suffix for ch in parameters['channels'].keys()]
    channel_locs = [parameters['channels'][ch] for ch in parameters['channels'].keys()]

    return all_channels, channel_locs


### USER EDITABLE FUNCTION WHERE ANALYSIS IS EXECUTED
def run_analysis(
    analysis_dispatch_inputs: AnalysisDispatchModel,
    analysis_parameters: SummaryPlotsAnalysisSpecification,
) -> dict | None:
    # run_analysis_jobs (in analysis_pipeline_utils) now owns building the
    # processing record, the docdb-already-exists check, and writing
    # results/metadata. This function only needs to run the analysis and
    # return a dict of output parameters.
    parameters = analysis_parameters.model_dump()

    curation = None
    if parameters['curation_csv'] is not None:
        try:
            curation, curation_full = data_curation_helpers.load_curation(parameters['curation_csv'])
        except Exception:
            logger.exception(f"Failed to load curation at {parameters['curation_csv']}. "
                             "Continuing without curation.")

    (df_sess, df_trials, df_events, df_fip) = r_utils.get_nwb_processed(
        analysis_dispatch_inputs.file_location, curation=curation, **parameters)



    ############## prepare computations for plotting ###############

    df_trials = analysis_utils.enrich_df_trials(df_trials)

    nwbs_all = r_utils.get_dummy_nwbs(df_trials, df_events, df_fip)
    ch_suffix = '' if (parameters['preprocessing'] == 'raw') else f"_{parameters['preprocessing']}"

    all_channels, channel_locs = get_all_channels(
        parameters, ch_suffix, df_fip if curation is not None else None)

    # plot summary plots
    plot_loc = '/results/plots/'

    if parameters["plot_save_format"] != "png":
        summary_plots.set_save_format(fmt=parameters["plot_save_format"])

    [Path(f"/results/data/{subject_id}").mkdir(parents=True, exist_ok=True) for subject_id in df_sess['subject_id'].unique()]

    # parameters = validate_pearsonr(parameters)


    # for pair in parameters['pearson_pairs']:

    #     signal1 = f"{pair[0]}{ch_suffix}"
    #     signal2 = f"{pair[1]}{ch_suffix}"

    #     nwbs_all = [analysis_utils.add_sliding_window_corr(
    #                 nwb,
    #                 signal1name=signal1,
    #                 signal2name=signal2,
    #             ) for nwb in nwbs_all]

    if {"rpe", "choice_split_rpe", "rpe_no_plots", "weekly", "bayer&glimcher"} & set(parameters["plot_types"]):

        offsets = [0.33,1]
        nwbs_by_week = r_utils.split_nwbs_by_week(nwbs_all)
        # TODO:
        # 1. [done] edit add_AUC_rpe_slope to A. save outside of analysis utils
        # B. go through it nwb by nwb, not week by week, have nwbs_by_week later
        # C. save data_column in combined_rpe_slope.
        # 2. need to run for data_norm and data_z_norm,
        # and save combined_rpe_slope with correct_name for columns
        # 3. then, in summary_plots, search for avg_signal with data_z or data accordingly (for averaged vs not)
        # NEXT, will need to an option to force data_z version
        (nwbs_by_week, combined_rpe_slope) = analysis_utils.add_AUC_and_rpe_slope(nwbs_by_week, all_channels,
                                                parameters["save_dfs"], data_column="data_z_norm", offsets=offsets)
        nwbs_all = [nwb for week in nwbs_by_week for nwb in week]

    ############## SAVE OR PREPARE PLOT_LOC ##############
    if not os.path.exists(plot_loc):
        os.makedirs(plot_loc)

    if parameters["save_dfs"] == True:
        r_utils.save_nwb_list(nwbs_all, '/results/data/', curation_full, df_sess)
    else:
        if curation_full is not None:
            suffix = "_".join(sorted(str(s) for s in df_sess['subject_id'].unique()))
            curation_path = Path("/results/data") / f"df_curation_{suffix}.csv"
            curation_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"saving curation to {curation_path}")
            curation_full.to_csv(curation_path, index=False)


    ############## RUN ANALYSIS ##############

    for channel, channel_loc in zip(all_channels, channel_locs):

        if "all_sess" in parameters["plot_types"]:
            logger.info("running NEURAL PSTH")
            summary_plots.plot_all_sess_PSTH(df_sess, nwbs_all, channel, channel_loc, loc = plot_loc)
            if "subsample" in parameters["plot_types"]:
                logger.info("running left/right subsampled NEURAL PSTH")
                nwb_subsample = [r_utils.subsample_lr_thirds(nwb) for nwb in nwbs_all]
                summary_plots.plot_all_sess_PSTH(df_sess, nwb_subsample, channel, channel_loc, loc = plot_loc + 'subsample_')

        if "rpe" in parameters["plot_types"]:
            logger.info("running NEURAL PSTH with RPE focus")
            summary_plots.plot_all_sess_RPE(df_sess, nwbs_all, channel, channel_loc, loc = plot_loc)
            if "subsample" in parameters["plot_types"]:
                logger.info("running left/right subsampled NEURAL PSTH with RPE focus")
                nwb_subsample = [r_utils.subsample_lr_thirds(nwb) for nwb in nwbs_all]
                summary_plots.plot_all_sess_RPE(df_sess, nwb_subsample, channel, channel_loc, loc = plot_loc + 'subsample_')

        if "choice_split_rpe" in parameters["plot_types"]:
            logger.info("running NEURAL PSTH with CHOICE SPLIT RPE")
            nwbs_all_split = [r_utils.split_nwb_by_choice(nwb) for nwb in nwbs_all]
            summary_plots.plot_all_sess_left_right_RPE_PSTH(df_sess, nwbs_all_split, channel, channel_loc, offsets, loc = plot_loc)

        if "sess_split_extra" in parameters["plot_types"]:
            logger.info("running hit/miss with early vs late")
            nwb_early, nwb_late = tuple(map(list, zip(*(r_utils.split_nwb_by_time(n) for n in nwbs_all))))
            summary_plots.plot_all_sess_PSTH_extras(df_sess, nwb_early, channel, channel_loc, loc = plot_loc + 'early_')
            summary_plots.plot_all_sess_PSTH_extras(df_sess, nwb_late, channel, channel_loc, loc = plot_loc + 'late_')



        if "all_sess_extra" in parameters["plot_types"]:
            summary_plots.plot_all_sess_PSTH_extras(df_sess, nwbs_all, channel, channel_loc, loc = plot_loc)

        if "bayer&glimcher" in parameters["plot_types"]:
            logger.info("running Bayer & Glimcher regression")
            _, _, bg_coef_df = summary_plots.plot_bayer_glimcher_rows(nwbs_all, channel, channel_loc, loc=plot_loc)
            if parameters["save_dfs"] and bg_coef_df is not None and len(bg_coef_df) > 0:
                channel_short = channel.split('_dff')[0] if '_dff' in channel else channel
                subject_id = df_sess['subject_id'].unique()[0]
                bg_coef_df.to_csv(f"/results/data/{subject_id}/bg_coef_{channel_short}_{channel_loc}.csv", index=False)

        if "weekly" in parameters["plot_types"]:
            logger.info("running weekly plots")

            summary_plots.plot_weekly_grid(df_sess, nwbs_by_week,combined_rpe_slope[combined_rpe_slope['channel'] == channel], channel, channel_loc, loc=plot_loc)


    if "behavior" in parameters["plot_types"]:
        logger.info("running ALL SESS behavior")
        if len(nwbs_all) > 5:
            nwb_batches = [nwbs_all[i:i+5] for i in range(0, len(nwbs_all), 5)]
            for nwb_batch in nwb_batches:
                summary_plots.plot_all_sess_behavior(df_sess, nwb_batch, loc = plot_loc)
    if "avg_lastN_sess" in parameters["plot_types"]:
        logger.info("running average last N sessions")
        summary_plots.plot_avg_final_N_sess(df_sess, nwbs_all, all_channels, channel_locs, final_N_sess = parameters["last_N_sess"], loc = plot_loc)

    return {}


# Most of the below code will not need to change per-analysis
# and will be moved to a shared library
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    run_analysis_jobs(
        analysis_input_model=SummaryPlotsAnalysisSpecification,
        analysis_output_model=SummaryPlotsAnalysisOutputs,
        run_function=run_analysis,
    )
