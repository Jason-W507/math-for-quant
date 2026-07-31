# SEC market-structure snapshot provenance

The numeric snapshot in `sec-order-placement-2014.json` is transcribed from the
U.S. Securities and Exchange Commission page “Equity Market Speed Relative to
Order Placement.” The page reports the five event shares and cancel-to-trade
ratios used here.

- Source: https://www.sec.gov/about/equity-market-speed-relative-order-placement-2014-02-market-structure
- Methodology: https://www.sec.gov/securities-topics/market-structure-analytics/market-activity-report-methodology
- Public-domain label used by the SEC open-data catalog: http://www.usa.gov/publicdomain/label/1.0/
- Registry marker: public domain
- Transformation: percentages were divided by 100; ratios were copied as
  reported. Values described by the SEC as approximate remain approximate.
- Selection: all five mutually exclusive placement categories shown in the
  source summary are retained; their event shares sum to one.

This aggregate snapshot contains no proprietary tick feed, order identifiers,
or personal information. It supports stress calibration and teaching examples;
it does not reconstruct an order book or establish causal effects.
