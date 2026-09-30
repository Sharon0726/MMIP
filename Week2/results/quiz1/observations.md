### 實測觀察
- **Logistic Regression**：門檻 0.45，Precision=0.8209、Recall=0.6077、F1=0.6984；FP=24、FN=71，漏報 FN較多。
- **Random Forest**：門檻 0.30，Precision=0.7751、Recall=0.8950、F1=0.8308；FP=47、FN=19，誤報 FP較多。
- 本次 Precision 較高的是 **Logistic Regression**（0.8209）。
- 本次 Recall 較高的是 **Random Forest**（0.8950）。
- 本次 F1 較高的是 **Random Forest**（0.8308）。
- 模擬資料的非線性結構可能影響兩模型差異；結果僅代表這次固定切分，不能推論所有資料都相同。
