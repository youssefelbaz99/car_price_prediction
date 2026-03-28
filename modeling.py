from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from helper import load_data, save_model, setup_logger
from config import MODEL_PATH
from preprocessing import clean_data
from enum_file import ConfigEnum

logger = setup_logger()

# 1. جلب البيانات وتنظيفها
df = clean_data(load_data(config.TRAIN_DATA))

X = df.drop(ConfigEnum.TARGET_COLUMN.value, axis=1)
y = df[ConfigEnum.TARGET_COLUMN.value]

# 2. فصل البيانات (Train & Val)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=ConfigEnum.TEST_SIZE.value, random_state=ConfigEnum.RANDOM_STATE.value
)

# 3. بناء النموذج (يمكنك تجربة أكثر من نموذج هنا)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 4. التحقق (Validation) وحساب الخطأ
predictions = model.predict(X_val)
error = mean_squared_error(y_val, predictions)
accuracy = r2_score(y_val, predictions)

# تسجيل النتائج في ملف الـ log
logger.info(f"Model: RandomForest | Error: {error} | R2 Score: {accuracy}")

# 5. حفظ أفضل نموذج
save_model(model, MODEL_PATH)