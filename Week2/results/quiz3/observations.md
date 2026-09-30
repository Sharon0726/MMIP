### 實測觀察
- Improved MLP AUC = **0.77227**；Random Forest AUC = **0.77608**。
- Random Forest − MLP 的 AUC 差值 = **+0.00381**；本次驗證集的整體 AUC 較高者為 **Random Forest**。
- AUC 較高表示這次驗證集的整體正負排序區分能力較佳；不代表每一個門檻的 Precision、Recall 都較佳。
- 固定 FPR 上限的 TPR 表提供局部操作區域的比較；實際門檻仍需依誤報／漏報代價另行決定。
- MLP 的 checkpoint 由本驗證集挑選；Random Forest 參數事先固定。這是作業的驗證比較，並非獨立測試或統計顯著性結論。
