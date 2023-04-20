# Show this arr_data to a table with columns(symbol, pnl_sum, win_rate_avg, periods)
from pandas import Timestamp


arr_data = [{'symbol': 'ARUSDT', 'pnl_sum': 55.18593409903845, 'win_rate_avg': 89.82091375985642, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 1.4603371115098343, 'win_rate': 84.61538461538461, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.7, 'trade_tick': 3, 'same_tick': 1.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8499999999999999}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': 58.73451553373067, 'win_rate': 91.17647058823529, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.8499999999999999, 'trade_tick': 3, 'same_tick': 3.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.6499999999999999}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': -5.008918546202057, 'win_rate': 93.67088607594937, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.75, 'trade_tick': 3, 'same_tick': 7.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.6499999999999999}}]}, {'symbol': 'AUDIOUSDT', 'pnl_sum': 95.12972929902048, 'win_rate_avg': 93.2159095522267, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 4.873231172865853, 'win_rate': 91.07142857142857, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.8499999999999999, 'trade_tick': 3, 'same_tick': 5.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8999999999999999}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': 29.99923524682077, 'win_rate': 93.47826086956522, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.5, 'trade_tick': 3, 'same_tick': 1.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8499999999999999}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 60.25726287933385, 'win_rate': 95.09803921568627, 'conditions': {'profit_percent': 0.5, 'atr_tick_multiplier': 0.75, 'trade_tick': 3, 'same_tick': 13.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.75}}]}, {'symbol': 'BANDUSDT', 'pnl_sum': 45.94775716562484, 'win_rate_avg': 91.50925925925925, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 14.401116584874252, 'win_rate': 93.0, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.6499999999999999, 'trade_tick': 3, 'same_tick': 6.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.55}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': 7.184893186296468, 'win_rate': 91.25, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.75, 'trade_tick': 3, 'same_tick': 13.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.45}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 24.361747394454124, 'win_rate': 90.27777777777779, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.7, 'trade_tick': 3, 'same_tick': 3.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8999999999999999}}]}, {'symbol': 'CELRUSDT', 'pnl_sum': 17.77898890069511, 'win_rate_avg': 88.4934669417428, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 18.840800315646465, 'win_rate': 88.88888888888889, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 4.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8999999999999999}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': -17.276365186990443, 'win_rate': 86.20689655172413, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.6499999999999999, 'trade_tick': 3, 'same_tick': 3.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8999999999999999}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 16.21455377203909, 'win_rate': 90.38461538461539, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.6499999999999999, 'trade_tick': 3, 'same_tick': 2.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.95}}]}, {'symbol': 'COMPUSDT', 'pnl_sum': -6.028372561983706, 'win_rate_avg': 86.4406779661017, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': -6.028372561983706, 'win_rate': 86.4406779661017, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.7, 'trade_tick': 3, 'same_tick': 4.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.5}}]}, {'symbol': 'DYDXUSDT', 'pnl_sum': 20.711129030745752, 'win_rate_avg': 90.31659765355418, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 30.503449230368968, 'win_rate': 95.65217391304348, 'conditions': {'profit_percent': 0.5, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 4.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8999999999999999}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': 51.888890250870624, 'win_rate': 85.71428571428571, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 5.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8999999999999999}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': -61.681210450493836, 'win_rate': 89.58333333333334, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.5, 'trade_tick': 3, 'same_tick': 4.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.75}}]}, {'symbol': 'INJUSDT', 'pnl_sum': 106.9617567262897, 'win_rate_avg': 90.9507149376331, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 44.97901829130509, 'win_rate': 95.0, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 4.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.75}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': 1.5787007930671462, 'win_rate': 84.21052631578947, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 7.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.7}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 60.404037641917476, 'win_rate': 93.64161849710982, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.5, 'trade_tick': 3, 'same_tick': 4.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.7}}]}, {'symbol': 'IOTXUSDT', 'pnl_sum': 28.71866021173777, 'win_rate_avg': 93.76625534497784, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 10.243187194749607, 'win_rate': 95.23809523809523, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 4.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.95}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': -19.9061059988124, 'win_rate': 88.70967741935483, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.55, 'trade_tick': 3, 'same_tick': 9.0, 'same_holding': 20, 'divide': 2, 'back_size': 0.45}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 38.38157901580056, 'win_rate': 97.35099337748345, 'conditions': {'profit_percent': 0.5, 'atr_tick_multiplier': 0.6499999999999999, 'trade_tick': 3, 'same_tick': 12.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.95}}]}, {'symbol': 'LINAUSDT', 'pnl_sum': 37.5973688824278, 'win_rate_avg': 90.8279323929343, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 5.698849042344428, 'win_rate': 95.34883720930233, 'conditions': {'profit_percent': 0.45, 'atr_tick_multiplier': 0.55, 'trade_tick': 3, 'same_tick': 2.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8499999999999999}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': 42.29489303255847, 'win_rate': 89.34426229508196, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.6499999999999999, 'trade_tick': 3, 'same_tick': 7.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8999999999999999}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': -10.396373192475092, 'win_rate': 87.79069767441861, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.5, 'trade_tick': 3, 'same_tick': 9.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.75}}]}, {'symbol': 'MASKUSDT', 'pnl_sum': -72.37840148606065, 'win_rate_avg': 92.90464830477843, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 38.89705496587237, 'win_rate': 96.55172413793103, 'conditions': {'profit_percent': 0.45, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 13.0, 'same_holding': 20, 'divide': 2, 'back_size': 0.95}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': -18.460214357192026, 'win_rate': 93.08176100628931, 'conditions': {'profit_percent': 0.5, 'atr_tick_multiplier': 0.5, 'trade_tick': 3, 'same_tick': 12.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.7}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': -92.815242094741, 'win_rate': 89.08045977011494, 'conditions': {'profit_percent': 0.5, 'atr_tick_multiplier': 0.45, 'trade_tick': 3, 'same_tick': 9.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.5}}]}, {'symbol': 'OCEANUSDT', 'pnl_sum': -145.47932384292193, 'win_rate_avg': 92.50981484484022, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': -7.940129379592927, 'win_rate': 96.96969696969697, 'conditions': {'profit_percent': 0.45, 'atr_tick_multiplier': 0.7, 'trade_tick': 3, 'same_tick': 2.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8999999999999999}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': -90.45320586603428, 'win_rate': 91.37055837563452, 'conditions': {'profit_percent': 0.45, 'atr_tick_multiplier': 0.45, 'trade_tick': 3, 'same_tick': 7.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8499999999999999}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': -47.0859885972947, 'win_rate': 89.1891891891892, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.8499999999999999, 'trade_tick': 3, 'same_tick': 13.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.95}}]}, {'symbol': 'OPUSDT', 'pnl_sum': -71.20973122427134, 'win_rate_avg': 91.42209271449137, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 33.053839218207884, 'win_rate': 88.52459016393442, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.7, 'trade_tick': 3, 'same_tick': 3.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8999999999999999}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': -42.43552380269048, 'win_rate': 91.17647058823529, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.45, 'trade_tick': 3, 'same_tick': 5.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.55}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': -61.828046639788745, 'win_rate': 94.56521739130434, 'conditions': {'profit_percent': 0.45, 'atr_tick_multiplier': 0.6499999999999999, 'trade_tick': 3, 'same_tick': 4.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.55}}]}, {'symbol': 'QTUMUSDT', 'pnl_sum': 81.28866044675806, 'win_rate_avg': 95.63063063063062, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 26.493481655771923, 'win_rate': 100.0, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 3.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.6499999999999999}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': 11.657821093123257, 'win_rate': 91.8918918918919, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 5.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.55}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 43.13735769786288, 'win_rate': 95.0, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.7, 'trade_tick': 3, 'same_tick': 12.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.95}}]}, {'symbol': 'RENUSDT', 'pnl_sum': 27.599357547453604, 'win_rate_avg': 91.93634134211383, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 33.17404146191679, 'win_rate': 95.78947368421052, 'conditions': {'profit_percent': 0.5, 'atr_tick_multiplier': 0.7, 'trade_tick': 3, 'same_tick': 6.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.95}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': -27.557590945948085, 'win_rate': 90.32258064516128, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.5, 'trade_tick': 3, 'same_tick': 3.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.95}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 21.982907031484903, 'win_rate': 89.6969696969697, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.5, 'trade_tick': 3, 'same_tick': 4.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8999999999999999}}]}, {'symbol': 'SANDUSDT', 'pnl_sum': 26.145891367501537, 'win_rate_avg': 92.53900709219859, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 16.391066017714657, 'win_rate': 100.0, 'conditions': {'profit_percent': 0.45, 'atr_tick_multiplier': 0.8499999999999999, 'trade_tick': 3, 'same_tick': 3.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.95}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': -3.4089004907545126, 'win_rate': 84.0, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.6499999999999999, 'trade_tick': 3, 'same_tick': 5.0, 'same_holding': 20, 'divide': 2, 'back_size': 0.55}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 13.163725840541392, 'win_rate': 93.61702127659575, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.6499999999999999, 'trade_tick': 3, 'same_tick': 5.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.95}}]}, {'symbol': 'SNXUSDT', 'pnl_sum': -41.189292345347674, 'win_rate_avg': 89.31232492997198, 'periods': [{'period_from': Timestamp('2022-12-31 00:00:00'), 'period_to': Timestamp('2023-01-30 00:00:00'), 'pnl': 1.8097464489993582, 'win_rate': 88.0, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 6.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8999999999999999}}, {'period_from': Timestamp('2023-01-30 00:00:00'), 'period_to': Timestamp('2023-03-01 00:00:00'), 'pnl': 21.596698986337596, 'win_rate': 87.5, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.7, 'trade_tick': 3, 'same_tick': 13.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.55}}, {'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': -64.59573778068463, 'win_rate': 92.43697478991596, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.6499999999999999, 'trade_tick': 3, 'same_tick': 12.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.55}}]}]

arr_data2 = [{'symbol': 'ARUSDT', 'pnl_sum': 45.43620990719624, 'win_rate_avg': 93.18181818181817, 'periods': [{'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 45.43620990719624, 'win_rate': 93.18181818181817, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.7, 'trade_tick': 3, 'same_tick': 2.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8499999999999999}}]}, {'symbol': 'AUDIOUSDT', 'pnl_sum': 52.50359197760849, 'win_rate_avg': 92.85714285714286, 'periods': [{'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 52.50359197760849, 'win_rate': 92.85714285714286, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 5.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.75}}]}, {'symbol': 'BANDUSDT', 'pnl_sum': 87.13229372246191, 'win_rate_avg': 92.0, 'periods': [{'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 87.13229372246191, 'win_rate': 92.0, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.6499999999999999, 'trade_tick': 3, 'same_tick': 6.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.55}}]}, {'symbol': 'COMPUSDT', 'pnl_sum': 9.915220603027901, 'win_rate_avg': 89.23076923076924, 'periods': [{'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 9.915220603027901, 'win_rate': 89.23076923076924, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 13.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.7}}]}, {'symbol': 'DYDXUSDT', 'pnl_sum': 0.971648863729588, 'win_rate_avg': 93.75, 'periods': [{'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 0.971648863729588, 'win_rate': 93.75, 'conditions': {'profit_percent': 0.5, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 4.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.8999999999999999}}]}, {'symbol': 'LDOUSDT', 'pnl_sum': 45.347594295048495, 'win_rate_avg': 91.54929577464789, 'periods': [{'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 45.347594295048495, 'win_rate': 91.54929577464789, 'conditions': {'profit_percent': 0.6, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 11.0, 'same_holding': 20, 'divide': 2, 'back_size': 0.95}}]}, {'symbol': 'LINAUSDT', 'pnl_sum': 4.6009895303277695, 'win_rate_avg': 89.55223880597015, 'periods': [{'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 4.6009895303277695, 'win_rate': 89.55223880597015, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 9.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.95}}]}, {'symbol': 'OPUSDT', 'pnl_sum': 6.162664439779419, 'win_rate_avg': 94.64285714285714, 'periods': [{'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 6.162664439779419, 'win_rate': 94.64285714285714, 'conditions': {'profit_percent': 0.45, 'atr_tick_multiplier': 0.7, 'trade_tick': 3, 'same_tick': 3.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.6499999999999999}}]}, {'symbol': 'QTUMUSDT', 'pnl_sum': 20.199318854353532, 'win_rate_avg': 95.83333333333334, 'periods': [{'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 20.199318854353532, 'win_rate': 95.83333333333334, 'conditions': {'profit_percent': 0.7000000000000001, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 3.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.6499999999999999}}]}, {'symbol': 'SNXUSDT', 'pnl_sum': 169.21124459539968, 'win_rate_avg': 97.82608695652173, 'periods': [{'period_from': Timestamp('2023-03-01 00:00:00'), 'period_to': Timestamp('2023-03-31 00:00:00'), 'pnl': 169.21124459539968, 'win_rate': 97.82608695652173, 'conditions': {'profit_percent': 0.65, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 6.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.6499999999999999}}]}]

for data in arr_data2:
    # print("symbol:", data['symbol'], " pnl_sum:", data['pnl_sum'])
    # convert data['periods'][0]['conditions'] object({'profit_percent': 0.65, 'atr_tick_multiplier': 0.8999999999999999, 'trade_tick': 3, 'same_tick': 6.0, 'same_holding': 20, 'divide': 4, 'back_size': 0.6499999999999999}) to string("0.65,0.8999999999999999,3,6.0,20,4,0.6499999999999999")
    conditions = data['periods'][0]['conditions']
    # convert string to float with round 2 decimal places if it's float
    condition_values = [x for x in conditions.values()]
    condition_values = [data['symbol'], data['pnl_sum'], data['win_rate_avg']] + condition_values
    condition_values = [str(round(float(x), 2)) if isinstance(x, float) else str(x) for x in condition_values]

    condition_string = ",".join(condition_values)
    print(condition_string)
