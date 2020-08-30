import umap
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import spearmanr, kendalltau
from const import double_division_tracks


def umap_transform(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    embeddings = fit.fit_transform(data)
    return embeddings, fit


def umap_plot(ax, embeddings, colors, n_components, hovertext=None, title='', colorbar=False, alpha=1.0, cmap='RdYlGn', linewidths=0):

    if n_components == 1:
        points = ax.scatter(embeddings, range(len(embeddings)), c=colors)
    if n_components == 2:
        points = ax.scatter(embeddings[:, 0], embeddings[:, 1], s=40, c=colors, cmap=cmap, alpha=alpha, linewidths=linewidths)
    if n_components == 3:
        ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, cmap=cmap)

    ax.set_title(title, fontsize=18)
    if colorbar:
        ax.colorbar(points)


def plot_distributions(*data, xlim=(4, 8.5), ylim=(5, 10)):
    n_plots = len(data)
    plt.figure(figsize=(5 * n_plots, 5))

    for i, (d, title) in enumerate(data):
        ax = plt.subplot(1, n_plots, i + 1)
        ax.scatter(d[:, 0], d[:, 1], 50, alpha=0.2, c='b')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title)
        ax.set_ylabel('Log Red')
        ax.set_xlabel('Log Green')

    plt.tight_layout()


def timelapse_single_frame_sync(df, tracks):
    plt.figure(figsize=(15, 8))
    fun = lambda x: x * 10 / 60

    d = {}
    for t in tracks:
        df_ = df[str(t)].dropna()
        x = df_['time'].values.flatten()
        y_g = df_[('green', 'median')].values
        y_r = df_[('red', 'median')].values
        g_mean_tl, r_mean_tl = df.loc[df_.index, ['gfp_frame_average', 'cy3_frame_average']].T.values

        y_g = y_g - g_mean_tl
        y_r = y_r - r_mean_tl
        plt.plot(fun(x), y_g, 'g.', fun(x), y_r, 'r.', alpha=0.01, linewidth=0)

        for x_, y_g_, y_r_ in zip(x, y_g, y_r):
            d.setdefault(x_, []).append((y_g_, y_r_))

    average_track = np.array([np.mean(d[k], 0) for k in sorted(d)])
    xm = fun(np.array(sorted(d)))
    [GFP_legend, Cy3_legend] = plt.plot(xm, average_track[:, 0], 'yellow', xm, average_track[:, 1], 'red', linewidth=5)
    plt.legend([GFP_legend, Cy3_legend], ['Mean GFP', 'Mean Cy3'], loc=1)

    plt.xlim((-33, 33))
    plt.ylim((-100, 2000))
    plt.xlabel('Time, h')
    plt.ylabel('Intensity')
    plt.title('Intensity of 1000 tracks synchronized on single division')
    plt.show()
    return average_track


def timelapse_double_frame_sync(df: pd.DataFrame, tracks: dict, nnods: int=500,
                                title: str='Intensity of 50 tracks synchronized on double division'):

    dd_frames = np.array(list(tracks.values()))
    average_track_duration = np.mean(dd_frames[:, 1] - dd_frames[:, 0]) * 10 / 60  # hours

    min_time = 0
    max_time = average_track_duration

    g = []
    r = []

    g_mean_tl, r_mean_tl = df[['gfp_frame_average', 'cy3_frame_average']].T.values
    xnew = np.linspace(min_time, max_time, nnods)

    fig, ax = plt.subplots(figsize=(15, 8))
    for t, frms in tracks.items():
        frames = df[('frame_num')].values.flatten()
        y_g = df[(str(t), 'green', 'median')].values
        y_r = df[(str(t), 'red', 'median')].values
        y_g = y_g - g_mean_tl
        y_r = y_r - r_mean_tl

        start = np.where(frames == frms[0])[0][0]
        stop = np.where(frames == frms[1])[0][0]

        y_g_ = y_g[start: stop + 1]
        y_r_ = y_r[start: stop + 1]
        x = np.linspace(min_time, max_time, len(y_g_))
        f_g = interpolate.interp1d(x, y_g_)
        f_r = interpolate.interp1d(x, y_r_)

        y_g_interpolated = f_g(xnew)
        y_r_interpolated = f_r(xnew)
        g.append(y_g_interpolated)
        r.append(y_r_interpolated)

        ax.plot(xnew, y_g_interpolated, 'g.', xnew, y_r_interpolated, 'r.', alpha=0.15, linewidth=0)

    g = np.array(g)
    r = np.array(r)
    g_median = np.median(g, axis=0)
    r_median = np.median(r, axis=0)

    [GFP_legend, Cy3_legend] = ax.plot(xnew, g_median, 'yellow', xnew, r_median, 'red', linewidth=5)
    ax.legend([GFP_legend, Cy3_legend], ['Median GFP', 'Median Cy3'], loc=1)
    ax.set_xlim((0, max_time))
    ax.set_ylim((-100, 2000))
    ax.set_xlabel('Time, h')
    ax.set_ylabel('Intensity')
    ax.set_title(title)
    # plt.show()
    return np.column_stack([g_median, r_median]), ax, [g, r]


def circular_tracking(intensities, embeddings, center_x=0, center_y=0, steps=50):

    s_ = embeddings - (center_x, center_y)
    angle = np.pi / 2 - np.arctan(s_[:, 1] / s_[:, 0])
    angle += np.pi * (s_[:, 0] < 0)
    angle *= 360 / (2 * np.pi)
    radius = np.sqrt(np.square(s_).sum(1))

    angle_intervals = np.digitize(angle, np.linspace(0, 360, steps))
    intensities_mean = []
    embeddings_mean = []
    for i in range(1, steps):
        intensities_mean.append(np.mean(intensities[angle_intervals == i], axis=0))
        embeddings_mean.append(np.mean(np.sqrt(np.square(s_[angle_intervals == i]).sum(1)), axis=0))

    intensities_mean = np.array(intensities_mean)
    embeddings_mean = np.array(embeddings_mean)
    angle_mean = np.linspace(0, 360, steps)

    return angle, radius, intensities_mean, embeddings_mean, angle_mean[:-1]


def normalize_intensities(intensities, angle, angle_shift, steps=100):
    angle_adjusted = angle + angle_shift
    angle_adjusted[angle_adjusted > 360] -= 360
    angle_adjusted[angle_adjusted < 0] += 360

    intervals = np.digitize(angle_adjusted, np.linspace(0, 360, steps))
    smoothed_track = []
    stdev = []
    quant = []
    moving = 5
    for i in range(1, steps):
        if i - moving < 1:
            rng = np.logical_or(steps - 1 + (i - moving) <= intervals, intervals <= i + moving)
        elif i + moving > steps - 1:
            rng = np.logical_or(i - moving <= intervals, intervals <= (i + moving) - steps + 1)
        else:
            rng = np.logical_and(i - moving <= intervals, intervals <= i + moving)
        smoothed_track.append(np.median(intensities[rng], axis=0))
        stdev.append(intensities[rng].std(axis=0))
        quant.append(len(intensities[rng]))
    smoothed_track = np.array(smoothed_track)
    stdev = np.array(stdev)

    lo = smoothed_track.min(axis=0)
    hi = smoothed_track.max(axis=0)
    normalized_track = (smoothed_track - lo) / (hi - lo)
    stdev = stdev / (hi - lo) * 0.5

    style = 'seaborn'
    plt.style.use(style)

    fig, ax1 = plt.subplots(figsize=(14, 12))
    div = 0.25
    linewidth = 2
    x = np.linspace(0, 360, steps - 1)
    ax1.plot(x, normalized_track[:, 0], c='green', linewidth=linewidth)
    ax1.fill_between(x=x, y1=normalized_track[:, 0] - stdev[:, 0], y2=normalized_track[:, 0] + stdev[:, 0],
                     color='green', edgecolor='', alpha=0.3)
    ax1.plot(x, normalized_track[:, 1], c='red', linewidth=linewidth)
    ax1.fill_between(x=x, y1=normalized_track[:, 1] - stdev[:, 1], y2=normalized_track[:, 1] + stdev[:, 1],
                     color='red', edgecolor='', alpha=0.3)

    ax1.tick_params('both', labelsize=14)
    ax1.set_xlabel('Angle, $^\circ$', fontsize=20)
    ax1.set_xticks(np.arange(0, 361, 36))
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylabel('Normalized Log Intensity', fontsize=20)
    ax1.yaxis.set_label_coords(-0.05, 0.6)
    l, h = (normalized_track - stdev).min(), (normalized_track + stdev).max()
    ax1.set_ylim((l - div * h) / (1 - div), h)
    ticks = ax1.get_yticks()
    ticks = [t if t >= l else '' for t in ticks]
    ax1.set_yticklabels(ticks)
    ax1.legend(['GFP: S/G2/M phases', 'Cy3: G1 phase'], loc='upper left', fontsize=20)

    ax2 = ax1.twinx()
    q, _, _ = ax2.hist(angle_adjusted, bins=100, density=True, color='b')

    ax2.tick_params('y', labelsize=14)
    ax2.set_ylabel('Normalized cell count', color='b', fontsize=20)
    ax2.yaxis.set_label_coords(1.05, 0.13)
    l, h = q.min(), q.max()
    ax2.set_ylim(0, h / div)
    ax2.tick_params('y', colors='b')
    ticks = ax2.get_yticks()
    ticks = [t if t <= h else '' for t in ticks]
    ax2.set_yticklabels(ticks)
    ax2.grid(False)

    ax1.set_title('Fluo signal with 0.5 std bands and cell count', fontsize=28)
    fig.tight_layout()
    return smoothed_track


def project_onto_fluo_plane(intensities, *tracks, log_const=300):
    sns.set(style='ticks', color_codes=True)
    sns.set_context('talk', font_scale=1.2, rc={'lines.linewidth': 3.0})
    linewidth = 10
    grid = sns.jointplot(x='Log Green', y='Log Red',
                         data=pd.DataFrame(data={"Log Green": intensities[:, 0], "Log Red": intensities[:, 1]}),
                         xlim=(5.2, 7.2), ylim=(5.4, 7.1),
                         color="violet",
                         space=0,
                         kind="hex",
                         height=18,
                         ratio=5,
                         joint_kws={'gridsize': 150},
                         marginal_kws={'color': 'green'})
    cbar_ax = grid.fig.add_axes([.85, .62, .03, .2])  # x, y, width, height
    plt.colorbar(cax=cbar_ax)
    plt.setp(grid.ax_marg_y.patches, color='r')

    for track, color, title in tracks:
        df = pd.DataFrame(data={'x': track[:, 0], 'y': track[:, 1]})
        grid.x = df.x
        grid.y = df.y
        grid.plot_joint(plt.plot, linestyle='--', linewidth=linewidth, c=color, label=title)

        # df = pd.DataFrame(data={'x': [track[-1, 0], track[0, 0]], 'y': [track[-1, 1], track[0, 1]]})
        # grid.x = df.x
        # grid.y = df.y
        # grid.plot_joint(plt.plot, linestyle='--', linewidth=linewidth, c=color)

    grid.fig.suptitle('Embedded trajectory vs. timelapse averages')
    plt.legend()
    plt.ylim((5.4, 6.6))
    plt.show()


def correlation_plot(df, df_index, som_bmu, gfp_key, cy3_key):
    def scale(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    def interp(x, y, out):
        return interpolate.interp1d(x, y)(out)

    def plot_trend(arr, color):
        mean = scale(np.mean(arr, axis=0))
        plt.plot(out * 100, mean, color=color, linewidth=3)
        plt.plot(out * 100,  mean + np.std(arr, axis=0), color=color, linewidth=1)
        plt.plot(out * 100, mean - np.std(arr, axis=0), color=color, linewidth=1)
        plt.title(f"Spearman:{spearmanr(mean, range(len(mean))).correlation: 0.2f},  Kendall's tau:{kendalltau(mean, range(len(mean))).correlation: 0.2f}")

    df['som'] = np.nan
    df.loc[df_index.flatten(), 'som'] = som_bmu

    division_indexes = sorted(double_division_tracks)
    interp_size = 100
    out = scale(np.array(range(interp_size)))
    red = np.zeros((len(division_indexes), interp_size))
    green = np.zeros_like(red)
    SOM_ = np.zeros_like(red)

    for i, key in enumerate(division_indexes):
        start, end = double_division_tracks[key]
        # Attention!! Added +3 to start and -1 to end as there were some imperfections in the division frame estimation
        inds = df.loc[(df['TRACK_ID'] == key) & (df['FRAME'] >= start + 3) & (df['FRAME'] <= (end - 1)), 'FRAME'].index
        time = scale(np.arange(len(inds)))  #time in hrs
        red[i, :] = interp(time, df.loc[inds, cy3_key].values, out) #red fluo
        green[i, :] = interp(time, df.loc[inds, gfp_key].values, out) #green fluo
        SOM_[i, :] = interp(time, df.loc[inds, 'som'].values, out) #predictor spring

    # Remove empty columns
    red, green, SOM_ = [scale(r[~np.all(r == 0, axis=1)]) for r in [red, green, SOM_]]

    plt.style.use('default')
    plt.figure(figsize=(10, 4))
    plot_trend(arr=SOM_, color=[0.2,0.2,0.2])
    plt.xlabel('Relative time between two divisions (%)', fontsize=15)
    plt.ylabel('Normalized value (A.U.)', fontsize=15)
    plt.tight_layout()


def correlation_plot_all(df, df_index, som_bmu):
    def scale(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    def interp(x, y, out):
        return interpolate.interp1d(x, y)(out)

    def plot_trend(arr, color):
        mean = scale(np.mean(arr, axis=0))
        plt.plot(mean, color=color, linewidth=3)
        plt.plot(mean + np.std(arr, axis=0), color=color, linewidth=1)
        plt.plot(mean - np.std(arr, axis=0), color=color, linewidth=1)
        plt.title(f"Spearman:{spearmanr(mean, range(len(mean))).correlation: 0.2f},  Kendall's tau:{kendalltau(mean, range(len(mean))).correlation: 0.2f}")

    df['som'] = np.nan
    df.loc[df_index.flatten(), 'som'] = som_bmu

    interp_size = 100
    track_indices = sorted(df.loc[df_index.flatten(), 'TRACK_ID'].unique())
    out = scale(np.array(range(interp_size)))
    SOM_ = []

    for i, key in enumerate(track_indices[::100]):
        som_ = df.loc[df['TRACK_ID'] == key, 'som'].values
        som_ = som_[~np.isnan(som_)]     # Remove empty values
        som_ = np.roll(som_, -np.argmin(som_)) # track sync on min som cluster number
#         som_ = som_[:np.argmax(som_)] # sync between min and max som cluster number, unfair
        if len(som_) < 10: continue
        time = scale(np.arange(len(som_))) # relative time
        som_ = interp(time, som_, out)
        SOM_.append(som_)

    SOM_ = scale(np.array(SOM_))

    plt.style.use('default')
    plt.figure(figsize=(10, 4))
    plot_trend(arr=SOM_, color=[0.2,0.2,0.2])
    plt.xlabel('Relative time (%)', fontsize=15)
    plt.ylabel('Normalized value (A.U.)', fontsize=15)
    plt.tight_layout()
