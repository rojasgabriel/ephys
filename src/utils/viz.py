import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from typing import Optional, Literal
from labdata.schema import (  # type: ignore
    Dataset,
    DatasetEvents,
    SpikeSorting,
    UnitMetrics,
    EphysRecording,
    UnitCount,
)
from spks.viz import plot_event_aligned_raster  # type: ignore
from spks.event_aligned import population_peth  # type: ignore
from scipy.stats import sem
import ipywidgets as widgets  # type: ignore
from IPython.display import display


class PSTHViewer:
    def __init__(
        self,
        subject: Optional[str] = None,
        session: Optional[str] = None,
        unit_criteria_id: int = 1,
        pre_seconds: float = 1,
        post_seconds: float = 1,
        binwidth_ms: int = 10,
        plot_type: Literal["raster", "heatmap", "psth"] = "raster",
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
    ) -> None:

        self.subject = subject
        self.session = session
        self.unit_criteria_id = unit_criteria_id
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.binwidth_ms = binwidth_ms
        self.plot_type = plot_type
        self.fig = figure
        self.ax = axes
        self._cax: Optional[Axes] = None  # dedicated axes for colorbar
        self._colorbar: Optional[Colorbar] = None

        # to be filled by compute()
        self.peth = None  # (units, trials, timebins)
        self.mean_peth = None  # (units, timebins)
        self.sem_peth = None  # (units, timebins)
        self.bin_centers = None  # (timebins,)

    def compute(self):
        """Run population_peth and cache results."""
        sess_query = (
            SpikeSorting()
            & f'subject_name = "{self.subject}"'
            & f'session_name = "{self.session}"'
        ).proj()

        good_unit_ids = (
            sess_query
            * (
                UnitCount.Unit
                & f"unit_criteria_id = {self.unit_criteria_id}"
                & "passes = 1"
            )
        ).fetch("subject_name", "session_name", "unit_id", as_dict=True)

        good_units = pd.DataFrame(
            ((SpikeSorting.Unit & good_unit_ids) * UnitMetrics).fetch(
                "unit_id", "spike_times", "depth", as_dict=True
            )
        )

        dset = (Dataset() & sess_query).proj()
        events = DatasetEvents.Digital() & dset
        events = pd.DataFrame(events.fetch_synced())

        # make sure this matches the mapping of inputs for the session of interest...
        align_ev = {
            "stim": events.query("event_name == '0'").event_timestamps.values[0],
            "trial_start": events.query("event_name == '2'").event_timestamps.values[0],
            "frames": events.query("event_name == '3'").event_timestamps.values[0],
            "left_port": events.query("event_name == '4'").event_timestamps.values[0],
            "center_port": events.query("event_name == '5'").event_timestamps.values[0],
            "right_port": events.query(
                "event_name == '6' & stream_name == 'obx'"
            ).event_timestamps.values[0],
        }

        stim = align_ev["stim"]
        stim_ev = np.concatenate([[stim[0]], stim[1:][np.diff(stim) > 0.025]])
        first_stim_ev = np.concatenate([[stim[0]], stim[1:][np.diff(stim) > 1]])
        align_ev.update({"stim_ev": stim_ev, "first_stim_ev": first_stim_ev})

        srate = float(
            (EphysRecording.ProbeSetting() & sess_query).fetch("sampling_rate")[0]
        )
        good_units = good_units.sort_values("depth", ascending=True)
        st_per_unit = {
            row["unit_id"]: row["spike_times"] / srate
            for _, row in good_units.iterrows()
        }

        return st_per_unit, align_ev

    def plot(
        self,
        spike_times: np.ndarray,
        event_times: np.ndarray,
        all_spike_times: Optional[dict] = None,
    ) -> None:
        """Plot raster, psth, or population heatmap onto self.ax."""
        if self.ax is None or self.fig is None:
            return
        self.ax.cla()
        # hide colorbar axes by default; heatmap will re-show it
        if self._cax is not None:
            self._cax.cla()
            self._cax.set_visible(False)
        binwidth_s = self.binwidth_ms / 1000.0

        if self.plot_type == "raster":
            plot_event_aligned_raster(
                event_times=event_times,
                spike_times=spike_times,
                pre_seconds=self.pre_seconds,  # type: ignore
                post_seconds=self.post_seconds,  # type: ignore
                ax=self.ax,
            )
            ymin, ymax = self.ax.get_ylim()
            self.ax.vlines(0, ymin, ymax, colors="r", linestyles="--")
            self.ax.set_xlabel("time from event (s)")
            self.ax.set_ylabel("trial")

        elif self.plot_type == "psth":
            peth, timebin_edges, _ = population_peth(
                all_spike_times=[spike_times],
                alignment_times=event_times,
                pre_seconds=self.pre_seconds,
                post_seconds=self.post_seconds,
                binwidth_ms=self.binwidth_ms,
                kernel=None,
                pad=0,
            )
            # peth returns spike counts per bin — divide by bin width to get sp/s
            # peth shape: (1, n_trials, n_timebins)
            mean_fr = np.mean(peth[0], axis=0) / binwidth_s
            sem_fr = sem(peth[0], axis=0) / binwidth_s
            bin_centers = (timebin_edges[:-1] + timebin_edges[1:]) / 2.0
            self.ax.plot(bin_centers, mean_fr, color="k")
            self.ax.fill_between(
                bin_centers,
                mean_fr - sem_fr,
                mean_fr + sem_fr,
                alpha=0.3,
                color="k",
            )
            ymin, ymax = self.ax.get_ylim()
            self.ax.vlines(0, ymin, ymax, colors="r", linestyles="--")
            self.ax.set_xlabel("time from event (s)")
            self.ax.set_ylabel("sp/s")

        elif self.plot_type == "heatmap":
            if all_spike_times is None:
                return
            unit_ids = list(all_spike_times.keys())
            peth, timebin_edges, _ = population_peth(
                all_spike_times=list(all_spike_times.values()),
                alignment_times=event_times,
                pre_seconds=self.pre_seconds,
                post_seconds=self.post_seconds,
                binwidth_ms=self.binwidth_ms,
                kernel=None,
                pad=0,
            )
            # peth shape: (n_units, n_trials, n_timebins)
            # average across trials and convert to sp/s
            pop_matrix = np.mean(peth, axis=1) / binwidth_s  # (n_units, n_timebins)
            n_units = pop_matrix.shape[0]
            im = self.ax.imshow(
                pop_matrix,
                aspect="auto",
                origin="upper",
                extent=(-self.pre_seconds, self.post_seconds, float(n_units), 0.0),
                cmap="afmhot_r",
            )
            self.ax.vlines(0, 0, n_units, colors="cyan", linestyles="--", linewidth=0.8)
            # label y-ticks with real unit IDs
            tick_step = max(1, n_units // 10)
            tick_indices = list(range(0, n_units, tick_step))
            self.ax.set_yticks([i + 0.5 for i in tick_indices])
            self.ax.set_yticklabels(
                [str(unit_ids[i]) for i in tick_indices], fontsize=7
            )
            self.ax.set_xlabel("time from event (s)")
            self.ax.set_ylabel("unit ID (sorted by depth)")
            if self._cax is not None:
                self._cax.set_visible(True)
                self._colorbar = self.fig.colorbar(im, cax=self._cax, label="sp/s")

        if self.fig is not None:
            self.fig.canvas.draw_idle()


class PSTHWidget:
    def __init__(self, viewer: PSTHViewer) -> None:
        self.viewer = viewer

        # fetch data once on init
        self.st_per_unit, self.align_ev = viewer.compute()

        # build widgets
        self.unit_slider = widgets.SelectionSlider(
            options=list(self.st_per_unit.keys()),
            value=list(self.st_per_unit.keys())[0],
            description="Unit ID:",
            continuous_update=False,
        )
        self.event_dropdown = widgets.Dropdown(
            options=list(self.align_ev.keys()),
            value="first_stim_ev",
            description="Event:",
        )
        self.plot_type_toggle = widgets.RadioButtons(
            options=["raster", "psth", "heatmap"],
            value=self.viewer.plot_type,
            description="Plot type:",
        )

        # wire up callbacks
        self.unit_slider.observe(self._update, names="value")
        self.event_dropdown.observe(self._update, names="value")
        self.plot_type_toggle.observe(self._on_plot_type_change, names="value")

    def _on_plot_type_change(self, change: object) -> None:
        self.viewer.plot_type = self.plot_type_toggle.value
        self.unit_slider.disabled = self.viewer.plot_type == "heatmap"
        self._update(None)

    def _update(self, change: object) -> None:
        unit_id = self.unit_slider.value
        event = self.event_dropdown.value
        self.viewer.plot(
            spike_times=self.st_per_unit[unit_id],
            event_times=self.align_ev[event],
            all_spike_times=self.st_per_unit,
        )

    def show(self) -> None:
        if self.viewer.fig is None:
            with plt.ioff():
                self.viewer.fig = plt.figure(figsize=(6, 5))
                gs = self.viewer.fig.add_gridspec(
                    1, 2, width_ratios=[20, 1], wspace=0.05
                )
                self.viewer.ax = self.viewer.fig.add_subplot(gs[0])
                self.viewer._cax = self.viewer.fig.add_subplot(gs[1])
                self.viewer._cax.set_visible(False)

        self._out = widgets.Output()
        with self._out:
            display(self.viewer.fig.canvas)

        display(
            widgets.VBox(
                [
                    widgets.HBox(
                        [self.unit_slider, self.event_dropdown, self.plot_type_toggle]
                    ),
                    self._out,
                ]
            )
        )
        self._update(None)
