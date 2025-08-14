from dataclasses import dataclass


@dataclass
class SlippageModel:
    """Simple model for estimating slippage in basis points.

    Parameters
    ----------
    hv_cap : float, optional
        Maximum allowed annualized volatility. Values of ``hv`` passed to
        :meth:`slippage_bp` will be clipped to the range ``[0, hv_cap]``.
    bp_per_hv : float, optional
        Number of basis points corresponding to ``hv=1.0``.
    """

    hv_cap: float = 1.0
    bp_per_hv: float = 10000.0

    def slippage_bp(self, hv: float) -> float:
        """Return slippage in basis points for a given volatility.

        Parameters
        ----------
        hv : float
            Annualized volatility (0~1).

        Returns
        -------
        float
            Slippage cost expressed in basis points.
        """
        hv = min(max(hv, 0.0), self.hv_cap)
        return hv * self.bp_per_hv
