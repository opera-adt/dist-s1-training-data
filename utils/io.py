import concurrent.futures
import functools
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, xy
from tqdm import tqdm


S1_START_DATE = pd.Timestamp('2014-01-01', tz='UTC')
EARLIEST_POST_DATE = pd.Timestamp('2024-06-01', tz='UTC')


class PandasJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # Let the base class default method raise the TypeError and handle other types
        return json.JSONEncoder.default(self, obj)


def serialize_rasterio_profile(profile: dict, out_path: Path) -> Path:
    profile_dict = dict(profile)
    if 'crs' in profile and profile['crs'] is not None:
        if isinstance(profile['crs'], CRS):
            profile_dict['crs'] = profile_dict['crs'].to_wkt()
    with Path.open(out_path, 'w') as f:
        json.dump(profile_dict, f, indent=4, cls=PandasJSONEncoder)
    return out_path


def deserialize_rasterio_profile(json_path: Path | str) -> rasterio.profiles.Profile:
    json_path = Path(json_path)
    with Path.open(json_path, 'r') as f:
        profile_dict = json.load(f)

    if isinstance(profile_dict.get('nodata'), str) and profile_dict['nodata'].lower() == 'nan':
        profile_dict['nodata'] = float('nan')

    if 'crs' in profile_dict and isinstance(profile_dict['crs'], str):
        profile_dict['crs'] = CRS.from_wkt(profile_dict['crs'])

    if 'transform' in profile_dict and isinstance(profile_dict['transform'], list):
        profile_dict['transform'] = Affine(
            profile_dict['transform'][0],
            profile_dict['transform'][1],
            profile_dict['transform'][2],
            profile_dict['transform'][3],
            profile_dict['transform'][4],
            profile_dict['transform'][5],
        )

    return rasterio.profiles.Profile(**profile_dict)


def write_one_tiff(arr: np.ndarray, profile: dict, out_path: Path):
    with rasterio.open(out_path, 'w', **profile) as ds:
        ds.write(arr, 1)


def extract_data_from_patch_directory(input_dir_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(input_dir_path / 'array_data.npz')
    ref_prof = deserialize_rasterio_profile(input_dir_path / 'ref_prof.json')
    with Path.open(input_dir_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    return data, ref_prof, metadata


def write_tiff_from_patch_directory(input_dir_path: Path | str, out_dir_path: Path | str):
    input_dir_path = Path(input_dir_path)
    out_dir_path = Path(out_dir_path)
    out_dir_path = out_dir_path / input_dir_path.name
    out_dir_path.mkdir(parents=True, exist_ok=True)
    data, ref_prof, metadata = extract_data_from_patch_directory(input_dir_path)

    pre_imgs = data['pre_imgs']
    post_img = data['post_img']

    copol = metadata['copol']
    crosspol = metadata['crosspol']

    n_pre_imgs = pre_imgs.shape[0]
    pre_ids = metadata['pre_ids']
    for i in range(n_pre_imgs):
        pre_img_copol = pre_imgs[i, 0, :, :].squeeze()
        pre_img_crosspol = pre_imgs[i, 1, :, :].squeeze()
        opera_id = pre_ids[i]
        pre_img_path_copol = out_dir_path / f'{opera_id}__{copol.upper()}_pre.tif'
        pre_img_path_crosspol = out_dir_path / f'{opera_id}__{crosspol.upper()}_pre.tif'
        write_one_tiff(pre_img_copol, ref_prof, pre_img_path_copol)
        write_one_tiff(pre_img_crosspol, ref_prof, pre_img_path_crosspol)

    post_id = metadata['post_id']
    post_img_path_copol = out_dir_path / f'{post_id}__{copol.upper()}_post.tif'
    post_img_path_crosspol = out_dir_path / f'{post_id}__{crosspol.upper()}_post.tif'
    write_one_tiff(post_img[0, :, :].squeeze(), ref_prof, post_img_path_copol)
    write_one_tiff(post_img[1, :, :].squeeze(), ref_prof, post_img_path_crosspol)
    return out_dir_path


def get_windowed_pre_imgs(
    df_burst_ts: gpd.GeoDataFrame, post_date: pd.Timestamp, n_samples: int = 7
) -> gpd.GeoDataFrame:
    df_pre_all = gpd.GeoDataFrame()
    for k in range(3):
        dt_anniversary = post_date - pd.Timedelta(days=(k + 1) * 365)
        df_pre = df_burst_ts[df_burst_ts.acq_dt <= dt_anniversary].tail(n_samples)
        if not df_pre.empty:
            df_pre_all = gpd.GeoDataFrame(pd.concat([df_pre_all, df_pre], axis=0))
    if not df_pre_all.empty:
        df_pre_all = df_pre_all.drop_duplicates(subset=['opera_id']).copy()
        df_pre_all = df_pre_all.sort_values(by=['acq_dt']).copy()
    return df_pre_all


def open_one(path: Path) -> tuple:
    with rasterio.open(path) as ds:
        X = ds.read(1)
        p = ds.profile
    return X, p


def write_one(X: np.ndarray, profile: dict, out_path: Path) -> Path:
    with rasterio.open(out_path, 'w', **profile) as ds:
        ds.write(X, 1)
    return out_path


def get_cropped_profile(profile: dict, slice_x: slice, slice_y: slice) -> dict:
    """
    Create a cropped profile from a reference profile and numpy slices.

    Parameters
    ----------
    profile : dict
        The reference rasterio profile.
    slice_x : slice
        The horizontal slice.
    slice_y : slice
        The vertical slice.

    Returns
    -------
    dict:
        The rasterio dictionary from cropping.
    """
    x_start = slice_x.start or 0
    y_start = slice_y.start or 0
    x_stop = slice_x.stop or profile['width']
    y_stop = slice_y.stop or profile['height']

    if (x_start < 0) | (x_stop < 0) | (y_start < 0) | (y_stop < 0):
        raise ValueError('Slices must be positive')

    width = x_stop - x_start
    height = y_stop - y_start

    profile_cropped = profile.copy()

    trans = profile['transform']
    x_cropped, y_cropped = xy(trans, y_start, x_start, offset='ul')
    trans_list = list(trans.to_gdal())
    trans_list[0] = x_cropped
    trans_list[3] = y_cropped
    tranform_cropped = Affine.from_gdal(*trans_list)
    profile_cropped['transform'] = tranform_cropped

    profile_cropped['height'] = height
    profile_cropped['width'] = width

    return profile_cropped


def serialize_one_patch(
    burst_id: str,
    patch_slice_dict: dict,
    arrs_copol_pre: list[np.ndarray],
    arrs_crosspol_pre: list[np.ndarray],
    arrs_copol_post: list[np.ndarray],
    arrs_crosspol_post: list[np.ndarray],
    sample_id: int,
    pre_ids: list[str],
    post_id: str,
    pre_dts: list[pd.Timestamp],
    post_dt: pd.Timestamp,
    ref_prof: dict,
    post_date_str: str,
    copol: str,
    crosspol: str,
    out_dir: Path,
) -> Path:
    sample_dir = out_dir / f'{burst_id}__{post_date_str}__{sample_id}'
    sample_dir.mkdir(parents=True, exist_ok=True)

    pre_year_float = [(dt - S1_START_DATE).total_seconds() / (365 * 60**2 * 24) for dt in pre_dts]
    post_year_float = [(dt - S1_START_DATE).total_seconds() / (365 * 60**2 * 24) for dt in [post_dt]]

    x_start, x_stop = patch_slice_dict['x_start'], patch_slice_dict['x_stop']
    y_start, y_stop = patch_slice_dict['y_start'], patch_slice_dict['y_stop']

    sy = np.s_[y_start:y_stop]
    sx = np.s_[x_start:x_stop]

    arrs_pre_copol_cropped = [arr_copol[sy, sx] for arr_copol in arrs_copol_pre]
    arrs_pre_crosspol_cropped = [arr_crosspol[sy, sx] for arr_crosspol in arrs_crosspol_pre]
    arrs_post_copol_cropped = [arr_copol[sy, sx] for arr_copol in arrs_copol_post]
    arrs_post_crosspol_cropped = [arr_crosspol[sy, sx] for arr_crosspol in arrs_crosspol_post]

    pre_imgs = np.stack(
        [
            np.stack([arr_copol, arr_crosspol], axis=0)
            for (arr_copol, arr_crosspol) in zip(arrs_pre_copol_cropped, arrs_pre_crosspol_cropped)
        ],
        axis=0,
    )

    post_img = np.concatenate(
        [
            np.stack([arr_copol, arr_crosspol], axis=0)
            for (arr_copol, arr_crosspol) in zip(arrs_post_copol_cropped, arrs_post_crosspol_cropped)
        ],
        axis=0,
    )

    npz_out_path = sample_dir / 'array_data.npz'
    acq_dts_float = pre_year_float + post_year_float
    np.savez(npz_out_path, pre_imgs=pre_imgs, post_img=post_img, acq_dts_float=acq_dts_float)

    metadata = {
        'pre_acq_dts': pre_dts,
        'post_acq_dt': post_dt,
        'post_id': post_id,
        'pre_ids': pre_ids,
        'copol': copol,
        'crosspol': crosspol,
    }
    out_path_json = sample_dir / 'metadata.json'
    with Path.open(out_path_json, 'w') as f:
        json.dump(metadata, f, indent=4, cls=PandasJSONEncoder)

    cropped_prof = get_cropped_profile(ref_prof, sx, sy)
    serialize_rasterio_profile(cropped_prof, sample_dir / 'ref_prof.json')

    return sample_dir


def generate_files_for_dataset_for_single_burst_id(
    df_burst_ts: gpd.GeoDataFrame | pd.DataFrame,
    patch_dir: Path,
    out_dir: Path = Path('dataset_samples_npz'),
    max_workers: int = 1,
) -> list[Path]:
    df_burst_ts = df_burst_ts.sort_values(by=['acq_dt']).reset_index(drop=True)
    burst_ids = df_burst_ts.jpl_burst_id.unique().tolist()
    assert len(burst_ids) == 1, 'Expected only one burst ID per dataset.'
    burst_id = burst_ids[0]

    post_indices_all_dates = df_burst_ts[df_burst_ts.acq_dt >= pd.Timestamp('2024-06-01', tz='UTC')].index.tolist()

    # Load all patch data for the burst
    patch_path = patch_dir / f'{burst_id}.parquet'
    all_patch_data = gpd.read_parquet(patch_path).to_dict('records')

    copol_paths_all = df_burst_ts.loc_path_copol
    crosspol_paths_all = df_burst_ts.loc_path_crosspol

    if 'VV.tif' in copol_paths_all[0]:
        copol = 'vv'
    elif 'HH.tif' in copol_paths_all[0]:
        copol = 'hh'
    else:
        raise ValueError(f'Unknown copol path: {copol_paths_all[0]}')
    crosspol = 'vh' if copol == 'vv' else 'hv'

    copol_data = [open_one(path) for path in tqdm(copol_paths_all, desc='Loading copol', disable=True)]
    arrs_copol_all, profiles_copol = zip(*copol_data)
    crosspol_data = [open_one(path) for path in tqdm(crosspol_paths_all, desc='Loading crosspol', disable=True)]
    arrs_crosspol_all, _ = zip(*crosspol_data)

    arrs_copol_all = np.stack(arrs_copol_all, axis=0)
    ref_prof = profiles_copol[0]

    burst_ts_inputs = []
    for post_index in tqdm(list(post_indices_all_dates), desc='post_dates', disable=True):
        post_indices = [post_index]
        df_post = df_burst_ts[df_burst_ts.index == post_index].copy()
        # Convert to dict with records orientation, explicitly typing for mypy
        post_records = [dict(r) for _, r in df_post.iterrows()]
        assert len(post_records) == 1, 'Expected only one post record per post date.'
        post_data = post_records[0]
        post_dt = pd.Timestamp(post_data['acq_dt'])
        post_date_str = post_data['acq_date_for_mgrs_pass']
        post_id = post_data['opera_id']

        # Convert DataFrame to GeoDataFrame if needed
        if isinstance(df_burst_ts, pd.DataFrame):
            df_burst_ts = gpd.GeoDataFrame(df_burst_ts)
        df_pre = get_windowed_pre_imgs(df_burst_ts, post_dt)
        if df_pre.empty:
            continue

        pre_indices = df_pre.index.tolist()
        pre_ids = df_pre.opera_id.tolist()
        pre_dts = df_pre.acq_dt.tolist()

        arrs_copol_pre = [arrs_copol_all[i] for i in pre_indices]
        arrs_crosspol_pre = [arrs_crosspol_all[i] for i in pre_indices]
        arrs_copol_post = [arrs_copol_all[i] for i in post_indices]
        arrs_crosspol_post = [arrs_crosspol_all[i] for i in post_indices]

        assert len(arrs_crosspol_post) == 1, 'Expected only one post array.'
        assert len(arrs_copol_post) == 1, 'Expected only one post array.'
        assert len(arrs_crosspol_pre) == len(pre_indices)
        assert len(arrs_copol_pre) == len(pre_indices)

        inputs_for_post_date = [
            {
                'burst_id': burst_id,
                'patch_slice_dict': patch_data,
                'arrs_copol_pre': arrs_copol_pre,
                'arrs_crosspol_pre': arrs_crosspol_pre,
                'arrs_copol_post': arrs_copol_post,
                'arrs_crosspol_post': arrs_crosspol_post,
                'pre_ids': pre_ids,
                'pre_dts': pre_dts,
                'post_id': post_id,
                'post_dt': post_dt,
                'post_date_str': post_date_str,
                'out_dir': out_dir,
                'sample_id': sample_id,
                'ref_prof': ref_prof,
                'copol': copol,
                'crosspol': crosspol,
            }
            for sample_id, patch_data in enumerate(all_patch_data)
        ]
        burst_ts_inputs.extend(inputs_for_post_date)

    if max_workers > 1:

        def serialize_one_patch_wrapper(input_dict):
            return serialize_one_patch(**input_dict)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(serialize_one_patch_wrapper, burst_ts_inputs),
                    total=len(burst_ts_inputs),
                    desc='serializing patches',
                    disable=True,
                )
            )
    else:
        results = [
            serialize_one_patch(**input_dict)
            for input_dict in tqdm(burst_ts_inputs, desc='serializing patches', disable=True)
        ]

    return results


def generate_files_for_dataset_wrapper(df_ts, patch_dir, out_dir, max_threads):
    return generate_files_for_dataset_for_single_burst_id(df_ts, patch_dir, out_dir=out_dir, max_workers=max_threads)


def generate_files_from_many_burst_dfs(
    df_ts_list: list[gpd.GeoDataFrame],
    patch_dir: Path,
    n_jobs: int = 1,
    max_threads: int = 1,
    out_dir: Path = Path('dataset_samples_npz'),
):
    if n_jobs > 1:
        # Create a partial function that's picklable
        partial_wrapper = functools.partial(
            generate_files_for_dataset_wrapper, patch_dir=patch_dir, out_dir=out_dir, max_threads=max_threads
        )

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(
                tqdm(
                    executor.map(partial_wrapper, df_ts_list),
                    total=len(df_ts_list),
                    desc='generating files',
                    disable=False,
                )
            )
    else:
        results = [
            generate_files_for_dataset_wrapper(df_ts, patch_dir, out_dir, max_threads)
            for df_ts in tqdm(df_ts_list, desc='generating files', disable=False)
        ]

    # Flatten the list of lists
    results = [item for sublist in results for item in sublist]
    return results


def get_npz_paths_for_dataset_from_one_burst_df(
    df_burst_ts: gpd.GeoDataFrame | pd.DataFrame,
    patch_dir: Path,
    data_dir: Path = Path('dataset_samples_npz'),
) -> list[Path]:
    df_burst_ts = df_burst_ts.sort_values(by=['acq_dt']).reset_index(drop=True)
    burst_ids = df_burst_ts.jpl_burst_id.unique().tolist()
    assert len(burst_ids) == 1, 'Expected only one burst ID per dataset.'
    burst_id = burst_ids[0]

    post_indices_all_dates = df_burst_ts[df_burst_ts.acq_dt >= pd.Timestamp('2024-06-01', tz='UTC')].index.tolist()

    # Load all patch data for the burst
    patch_path = patch_dir / f'{burst_id}.parquet'
    all_patch_data = gpd.read_parquet(patch_path).to_dict('records')

    out_dirs_list = []
    for post_index in tqdm(list(post_indices_all_dates), desc='post_dates', disable=True):
        df_post = df_burst_ts[df_burst_ts.index == post_index].copy()
        post_records = [dict(r) for _, r in df_post.iterrows()]
        assert len(post_records) == 1, 'Expected only one post record per post date.'
        post_data = post_records[0]
        post_dt = pd.Timestamp(post_data['acq_dt'])
        post_date_str = post_data['acq_date_for_mgrs_pass']

        # Convert DataFrame to GeoDataFrame if needed
        if isinstance(df_burst_ts, pd.DataFrame):
            df_burst_ts = gpd.GeoDataFrame(df_burst_ts)
        df_pre = get_windowed_pre_imgs(df_burst_ts, post_dt)
        if df_pre.empty:
            continue

        out_dirs_list.extend(
            [
                data_dir / f'{burst_id}__{post_date_str}__{sample_id}'
                for sample_id, patch_data in enumerate(all_patch_data)
            ]
        )
    npz_paths_list = [out_dir / 'array_data.npz' for out_dir in out_dirs_list if (out_dir / 'array_data.npz').exists()]
    return npz_paths_list


def visualize_patch_sample_directory(input_dir_path: Path | str, max_pre_images: int | None = None):
    input_dir_path = Path(input_dir_path)
    data, ref_prof, metadata = extract_data_from_patch_directory(input_dir_path)
    pre_imgs = data['pre_imgs']
    post_img = data['post_img']
    copol = metadata['copol']
    crosspol = metadata['crosspol']

    n_pre = pre_imgs.shape[0]
    time_indices = list(reversed(range(n_pre)))
    if max_pre_images is not None:
        time_indices = time_indices[:max_pre_images]
    n_images = max(len(time_indices), 1)
    fig, axes = plt.subplots(n_images, 4, figsize=(20, 5 * max(n_images, 1)))

    pre_dates = metadata['pre_acq_dts']
    post_date = metadata['post_acq_dt']

    if n_pre == 1:
        axes = axes.reshape(1, -1)

    # Plot pre-images
    vmin_copol = np.nanpercentile(pre_imgs[:, 0, :, :], 2)
    vmax_copol = np.nanpercentile(pre_imgs[:, 0, :, :], 98)
    vmin_crosspol = np.nanpercentile(pre_imgs[:, 1, :, :], 2)
    vmax_crosspol = np.nanpercentile(pre_imgs[:, 1, :, :], 98)

    for ax_index, i in enumerate(time_indices):
        # Plot copol pre-image
        axes[ax_index, 0].imshow(pre_imgs[i, 0, :, :], cmap='gray', vmin=vmin_copol, vmax=vmax_copol)
        axes[ax_index, 0].set_title(f'{pre_dates[i]} {copol.upper()}')
        axes[ax_index, 0].axis('off')

        # Plot crosspol pre-image
        axes[ax_index, 1].imshow(pre_imgs[i, 1, :, :], cmap='gray', vmin=vmin_crosspol, vmax=vmax_crosspol)
        axes[ax_index, 1].set_title(f'{pre_dates[i]} {crosspol.upper()}')
        axes[ax_index, 1].axis('off')

    # Add column titles
    fig.text(0.25, 1.0, 'Pre-images', ha='center', va='bottom', fontsize=12)
    fig.text(0.75, 1.0, 'Post-image', ha='center', va='bottom', fontsize=12)

    # Plot post-images in first row
    mid_row = 0

    # Plot copol post-image
    axes[mid_row, 2].imshow(post_img[0, :, :], cmap='gray', vmin=vmin_copol, vmax=vmax_copol)
    axes[mid_row, 2].set_title(f'{post_date} {copol.upper()}')
    axes[mid_row, 2].axis('off')

    # Plot crosspol post-image
    axes[mid_row, 3].imshow(post_img[1, :, :], cmap='gray', vmin=vmin_crosspol, vmax=vmax_crosspol)
    axes[mid_row, 3].set_title(f'{post_date} {crosspol.upper()}')
    axes[mid_row, 3].axis('off')

    # Hide unused axes
    for i in range(n_images):
        if i != mid_row:
            axes[i, 2].axis('off')
            axes[i, 3].axis('off')
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            axes[i, 3].set_xticks([])
            axes[i, 3].set_yticks([])

    plt.tight_layout()
    return fig


def serialize_npz_paths(
    data: list,
    out_dir_parquet: Path | str,
    patch_dir: Path,
    sample_dataset_dir: Path | str,
    save_every: int = 100,
):
    out_dir = Path(out_dir_parquet)
    out_dir.mkdir(exist_ok=True, parents=True)

    sample_dataset_dir = Path(sample_dataset_dir)
    assert sample_dataset_dir.exists(), f'Sample dataset directory does not exist: {sample_dataset_dir}'

    npz_path_list = []
    n = len(data)

    for k, df in tqdm(enumerate(data), total=n):
        npz_path_list.extend(get_npz_paths_for_dataset_from_one_burst_df(df, patch_dir, sample_dataset_dir))

        if k and (k % save_every == 0):
            df_out = pd.DataFrame({'npz_path': list(map(str, npz_path_list))})
            out_path = out_dir / f'npz_path_{k}__{n}.parquet'
            df_out.to_parquet(out_path)
            npz_path_list = []

    if npz_path_list:
        df_out = pd.DataFrame({'npz_path': list(map(str, npz_path_list))})
        out_path = out_dir / f'npz_path_{k}__{n}.parquet'
        df_out.to_parquet(out_path)

    return out_dir_parquet
