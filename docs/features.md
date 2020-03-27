## Features

特征和label拉成一个csv文件，如果样本数量太多的话可以考虑每4096个样本放在一个csv里面，csv的特征组排列顺序如下

特征 | 维度 | 说明 |
--- | :--: | :--: | 
day_1_highest | M | 第一天最高价的航班特征抽取 |
day_1_lowest | M | 第一天最低价的航班特征抽取 |
day_1_middle | M | 第一天价格位于中间值的航班特征抽取 |
day_1_global | K | 第一天的共享特征，如航班总量，价格均值方差等 |
day_2_highest | M | 第二天最高价 |
day_2_lowest | M | 第二天最低价 |
day_2_middle | M | 第二天中间值航班 |
day_2_global | K | 第二天的共享特征 |
... | ... | ... |
day_7_highest | M | 天数暂定总长度7天 |
day_7_lowest | M | 天数暂定总长度7天 |
day_7_middle | M | 天数暂定总长度7天 |
day_7_global | K | 天数暂定总长度7天 |
global_feature | N | 共享特征，如需要预测之后第几天的出价、出价当天是否为节假日/双休日、是否有特殊政策、航空公司信息，等等 |
label | 1 | 标签，最终实际价格 |

## Network

day_