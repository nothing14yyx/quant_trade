import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data_loader import DataLoader
from .utils.helper import calc_features_raw
from .utils.robust_scaler import compute_robust_z_params, save_scaler_params_to_json

logger = logging.getLogger(__name__)


def generate_offline_price_table(
    dl: DataLoader | None = None,
    symbols: Iterable[str] | None = None,
    intervals: Iterable[str] = ("1h", "4h", "d1"),
    out_dir: str | Path = "data/offline_prices",
) -> None:
    """导出多周期价位表并计算 RobustScaler 参数.

    Parameters
    ----------
    dl : DataLoader, optional
        若未指定则自动创建 ``DataLoader`` 实例。
    symbols : iterable of str, optional
        要导出的币种列表，默认为 ``dl.get_top_symbols(dl.topn)``。
    intervals : iterable of str, default ("1h","4h","d1")
        价位表所需的周期列表。
    out_dir : str or Path, default "data/offline_prices"
        CSV 文件与缩放参数的保存目录。
    """

    dl = dl or DataLoader()
    if symbols is None:
        symbols = dl.get_top_symbols(dl.topn)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    all_params: dict[str, dict] = {}
    for sym in symbols:
        parts = []
        for iv in intervals:
            df = dl.load_klines_db(sym, iv)
            if df is None or df.empty:
                continue
            feats = calc_features_raw(df, iv)
            feats = feats.reset_index()
            feats["symbol"] = sym
            feats["interval"] = iv
            parts.append(feats)
        if not parts:
            continue
        df_all = pd.concat(parts, ignore_index=True)
        df_all.to_csv(out_dir_path / f"{sym}.csv", index=False)
        numeric_cols = df_all.select_dtypes("number").columns.tolist()
        params = compute_robust_z_params(df_all, numeric_cols)
        all_params[sym] = params
        logger.info("%s exported with %s rows", sym, len(df_all))

    if all_params:
        scaler_path = out_dir_path / "price_scaler.json"
        save_scaler_params_to_json(all_params, str(scaler_path))
        logger.info("Scaler params saved to %s", scaler_path)
    else:
        logger.warning("no data exported, scaler params not saved")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_offline_price_table()
