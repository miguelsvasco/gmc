import os
import torch
import numpy as np
import logging.config
from pytorch_lightning import LightningModule
from gmc_code.DelaunayComponentAnalysis.schemes import (
    DCALoggers,
    DelaunayGraphParams,
    ExperimentDirs,
    GeomCAParams,
    HDBSCANParams,
    REData,
)
from gmc_code.DelaunayComponentAnalysis.DCA import DelaunayComponentAnalysis
import gmc_code.DelaunayComponentAnalysis._DCA_visualization as DCA_visualization


class DCAEvaluator(LightningModule):
    def __init__(
        self,
        model,
        scenario,
        machine_path,
        minimum_cluster_size,
        unique_modality_idxs,
        unique_modality_dims,
        partial_modalities_idxs,
    ):
        super(DCAEvaluator, self).__init__()

        self.model = model
        self.model.eval()
        self.scenario = scenario
        self.mcs = minimum_cluster_size
        self.unique_modality_idxs = unique_modality_idxs
        self.unique_modality_dims = unique_modality_dims
        self.partial_modalities_idxs = partial_modalities_idxs
        self.machine_path = machine_path

    # ---
    # --- Pytorch ligthning --- #
    def test_step(self, batch, batch_idx):
        data = batch[0]
        output_dict = {}

        # Forward pass through the encoder to get representations
        batch_R_repr = self.model.encode(data, sample=False)
        output_dict[-1] = batch_R_repr

        # Drop modalities
        for k in range(len(data)):
            E_data = [None if k != j else data[k] for j in range(len(data))]
            batch_E_repr = self.model.encode(E_data, sample=False)
            output_dict[k] = batch_E_repr
        return output_dict

    def evaluate(self, R, E, experiment_id):

        # initialize DCA params from ingredients
        data_config = REData(
            R=R, E=E, input_array_dir=os.path.join(experiment_id, "logs")
        )

        experiment_config = ExperimentDirs(
            experiment_dir=self.trainer.default_root_dir,
            experiment_id=experiment_id,
            precomputed_folder=os.path.join(experiment_id, "logs"),
        )

        graph_config = DelaunayGraphParams(
            executable_filepath=self.machine_path,
            unfiltered_edges_dir=os.path.join(experiment_id, "logs"),
            filtered_edges_dir=os.path.join(experiment_id, "logs"),
        )
        hdbscan_config = HDBSCANParams(
            min_cluster_size=self.mcs, clusterer_dir=os.path.join(experiment_id, "logs")
        )
        geomCA_config = GeomCAParams()
        exp_loggers = DCALoggers(experiment_config.DCA_dir)

        logging.config.dictConfig(exp_loggers.loggers)
        logger = logging.getLogger("experiment_logger")
        logger.info("Starting to run DCA...")

        DCA_algorithm = DelaunayComponentAnalysis(
            experiment_config,
            graph_config,
            hdbscan_config,
            geomCA_config,
            version=exp_loggers.version,
        )

        # Evaluate DCA
        Delaunay_graph = DCA_algorithm.fit(data_config)

        # Plot results
        DCA_visualization._plot_UMAP_components(
            Delaunay_graph,
            experiment_id,
            os.path.join(self.trainer.default_root_dir, data_config.input_array_filepath),
            os.path.join(self.trainer.default_root_dir, hdbscan_config.input_array_labels_filepath),
            DCA_algorithm.visualization_dir,
        )

        DCA_visualization._plot_RE_components_consistency(
            Delaunay_graph,
            DCA_algorithm.visualization_dir,
            min_comp_size=2,
            annotate_largest=True,
            display_smaller=False,
        )

        DCA_visualization._plot_RE_components_quality(
            Delaunay_graph,
            DCA_algorithm.visualization_dir,
            min_comp_size=2,
            annotate_largest=True,
            display_smaller=False,
        )

        # Extract metrics
        for stat, stat_value in Delaunay_graph.network_stats.__dict__.items():
            print(" ====> " + f"{stat}: {stat_value}")

        return Delaunay_graph

    def log_Delaunay_graph_stats(self, Delaunay_graph, dca_experiment_id):
        self.logger.log_metric(
            f"{dca_experiment_id}_P", Delaunay_graph.network_stats.precision
        )
        self.logger.log_metric(
            f"{dca_experiment_id}_R", Delaunay_graph.network_stats.recall
        )
        self.logger.log_metric(
            f"{dca_experiment_id}_q", Delaunay_graph.network_stats.network_quality
        )
        self.logger.log_metric(
            f"{dca_experiment_id}_c", Delaunay_graph.network_stats.network_consistency
        )

        # Component stats
        for comp_idx in range(Delaunay_graph.first_trivial_component_idx):
            self.logger.log_metric(
                f"{dca_experiment_id}_component{comp_idx}_consistency",
                Delaunay_graph.comp_stats[comp_idx].comp_consistency,
            )
            self.logger.log_metric(
                f"{dca_experiment_id}_component{comp_idx}_quality",
                Delaunay_graph.comp_stats[comp_idx].comp_quality,
            )
            self.logger.log_metric(
                f"{dca_experiment_id}_component{comp_idx}_num_edges",
                Delaunay_graph.comp_stats[comp_idx].num_total_comp_edges,
            )
            self.logger.log_metric(
                f"{dca_experiment_id}_component{comp_idx}_num_RE_edges",
                Delaunay_graph.comp_stats[comp_idx].num_comp_RE_edges,
            )
            self.logger.log_metric(
                f"{dca_experiment_id}_component{comp_idx}_num_R",
                len(Delaunay_graph.comp_stats[comp_idx].Ridx),
            )
            self.logger.log_metric(
                f"{dca_experiment_id}_component{comp_idx}_num_E",
                len(Delaunay_graph.comp_stats[comp_idx].Eidx),
            )

        # Network stats vizualization
        self.logger.log_artifact(
            name=f"{dca_experiment_id}_comp_consistency.png",
            filepath=os.path.join(
                self.trainer.default_root_dir,
                dca_experiment_id,
                "visualization",
                "component_consistency_min_size2_annotated1_displaysmaller0.png",
            ),
        )
        self.logger.log_artifact(
            name=f"{dca_experiment_id}_comp_quality.png",
            filepath=os.path.join(
                self.trainer.default_root_dir,
                dca_experiment_id,
                "visualization",
                "component_quality_min_size2_annotated1_displaysmaller0.png",
            ),
        )
        self.logger.log_artifact(
            name=f"{dca_experiment_id}_all_RE.png",
            filepath=os.path.join(
                self.trainer.default_root_dir,
                dca_experiment_id,
                "visualization",
                f"{dca_experiment_id}_all_RE.png",
            ),
        )
        self.logger.log_artifact(
            name=f"{dca_experiment_id}_all.png",
            filepath=os.path.join(
                self.trainer.default_root_dir,
                dca_experiment_id,
                "visualization",
                f"{dca_experiment_id}_all.png",
            ),
        )
        # Add result files
        for filename in os.listdir(
            os.path.join(
                self.trainer.default_root_dir, dca_experiment_id, "visualization"
            )
        ):
            if "_comp" in filename:
                self.logger.log_artifact(
                    name=f"{dca_experiment_id}_{filename}",
                    filepath=os.path.join(
                        self.trainer.default_root_dir,
                        dca_experiment_id,
                        "visualization",
                        filename,
                    ),
                )

        # Add result files
        for filename in os.listdir(
            os.path.join(self.trainer.default_root_dir, dca_experiment_id)
        ):
            if "results_version" in filename:
                self.logger.log_artifact(
                    name=f"{dca_experiment_id}_{filename}",
                    filepath=os.path.join(
                        self.trainer.default_root_dir, dca_experiment_id, filename
                    ),
                )
        self.logger.log_artifact(
            name=f"{dca_experiment_id}_DCA_input_version0.json",
            filepath=os.path.join(
                self.trainer.default_root_dir,
                dca_experiment_id,
                "DCAInput_version0.json",
            ),
        )

    def test_epoch_end(self, outputs):
        n_mod = len(list(outputs[0].keys()))
        R = torch.concat([outputs[i][-1] for i in range(len(outputs))]).cpu().numpy()

        E_repr = []
        for mod in range(n_mod - 1):
            E = (
                torch.concat([outputs[i][mod] for i in range(len(outputs))])
                .cpu()
                .numpy()
            )
            if mod in self.unique_modality_idxs:
                E = np.unique(E.round(4), axis=0)
                assert (
                    E.shape[0]
                    == self.unique_modality_dims[self.unique_modality_idxs.index(mod)]
                )
            E_repr.append(E)
            Delaunay_graph = self.evaluate(R, E, f"joint_m{mod}")
            self.log_Delaunay_graph_stats(Delaunay_graph, f"joint_m{mod}")
            del Delaunay_graph, E

        # Extra eval
        for mod0, mod1 in self.partial_modalities_idxs:
            Delaunay_graph = self.evaluate(
                E_repr[mod0], E_repr[mod1], f"m{mod0}_m{mod1}"
            )
            self.log_Delaunay_graph_stats(Delaunay_graph, f"m{mod0}_m{mod1}")

        num_halv_R = int(R.shape[0] / 2)
        Delaunay_graph = self.evaluate(R[:num_halv_R], R[num_halv_R:], f"joint_joint")
        self.log_Delaunay_graph_stats(Delaunay_graph, f"joint_joint")

