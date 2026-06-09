# Retail Black Friday Sales Dataset Explanation

## Dataset Overview

### English

This dataset contains 100,000 retail transaction records related to Black Friday and promotional sales activity. Each row represents a single purchase transaction, including customer demographics, location, product information, price, discount, quantity, payment method, purchase date, purchase hour, weekend flag, and Black Friday flag.

The dataset is useful for exploratory data analysis, customer behavior analysis, product category performance analysis, discount effectiveness analysis, sales trend analysis, and basic machine learning workflows related to retail transactions.

### Bahasa Indonesia

Dataset ini berisi 100.000 data transaksi retail yang berkaitan dengan Black Friday dan aktivitas promosi penjualan. Setiap baris merepresentasikan satu transaksi pembelian, dengan informasi demografi pelanggan, lokasi, produk, harga, diskon, jumlah barang, metode pembayaran, tanggal pembelian, jam pembelian, penanda akhir pekan, dan penanda Black Friday.

Dataset ini cocok digunakan untuk exploratory data analysis, analisis perilaku pelanggan, analisis performa kategori produk, analisis efektivitas diskon, analisis tren penjualan, serta workflow machine learning dasar yang berkaitan dengan transaksi retail.

## Dataset Structure

- Rows: 100,000
- Columns: 18
- Time range: 2025-11-24 to 2025-12-01
- Granularity: transaction-level data
- Main transaction identifier: `transaction_id`
- Main customer identifier: `customer_id`
- Main product identifier: `product_id`
- Main date column: `purchase_date`
- Main revenue metric: `purchase_amount`
- Black Friday date: 2025-11-28

## Column Explanation

| Column | Data Type | English Explanation | Penjelasan Bahasa Indonesia | Analysis Notes |
| --- | --- | --- | --- | --- |
| `transaction_id` | String | Unique identifier for each transaction. | ID unik untuk setiap transaksi. | Useful for checking duplicate transactions and transaction counts. |
| `customer_id` | String | Unique identifier for each customer. | ID unik untuk setiap pelanggan. | Useful for customer-level analysis, repeat purchase behavior, and segmentation. |
| `age_group` | String | Customer age group category. | Kelompok usia pelanggan. | Useful for comparing purchasing behavior across age groups. |
| `gender` | String | Customer gender category. | Kategori gender pelanggan. | Useful for demographic analysis, but should be interpreted carefully and ethically. |
| `city` | String | City where the purchase was made or associated with the customer. | Kota tempat transaksi terjadi atau kota yang terkait dengan pelanggan. | Useful for location-based sales comparison. |
| `customer_segment` | String | Customer segment label such as New, Returning, Loyal, or VIP. | Segmentasi pelanggan seperti New, Returning, Loyal, atau VIP. | Useful for analyzing revenue contribution and purchasing behavior by customer type. |
| `product_id` | String | Unique identifier for the purchased product. | ID unik untuk produk yang dibeli. | Useful for product-level popularity and performance analysis. |
| `product_category` | String | Category of the purchased product. | Kategori produk yang dibeli. | Useful for category-level sales, revenue, and discount analysis. |
| `original_price` | Float | Product price before discount. | Harga produk sebelum diskon. | Can be compared with `final_price` to analyze discount impact. |
| `discount_pct` | Integer | Discount percentage applied to the product. | Persentase diskon yang diberikan pada produk. | Useful for studying discount distribution and discount effectiveness. |
| `final_price` | Float | Product price after discount. | Harga produk setelah diskon. | Calculated from original price and discount percentage. |
| `quantity` | Integer | Number of units purchased in the transaction. | Jumlah unit produk yang dibeli dalam transaksi. | Useful for basket size and item volume analysis. |
| `purchase_amount` | Float | Total transaction value after discount and quantity. | Total nilai transaksi setelah diskon dan jumlah unit. | Main revenue metric. Usually calculated as `final_price * quantity`. |
| `payment_method` | String | Payment method used for the transaction. | Metode pembayaran yang digunakan dalam transaksi. | Useful for payment preference analysis. |
| `purchase_date` | String/Date | Date when the transaction occurred. | Tanggal terjadinya transaksi. | Convert to datetime for time series analysis. |
| `purchase_hour` | Integer | Hour of the day when the transaction occurred, from 0 to 23. | Jam transaksi terjadi, dari 0 sampai 23. | Useful for hourly sales pattern analysis. |
| `is_weekend` | Integer/Boolean | Indicates whether the transaction happened on a weekend. | Menunjukkan apakah transaksi terjadi pada akhir pekan. | Useful for comparing weekday and weekend purchasing behavior. |
| `is_black_friday` | Integer/Boolean | Indicates whether the transaction happened on Black Friday. | Menunjukkan apakah transaksi terjadi pada Black Friday. | Key flag for comparing Black Friday vs non-Black Friday performance. |

## Suggested Analysis Questions

### English

1. How many transactions occurred during the observation period?
2. How much total revenue was generated during the Black Friday sales period?
3. How does Black Friday revenue compare with non-Black Friday revenue?
4. Which product categories generated the highest revenue?
5. Which product categories had the highest average discount?
6. Which customer segments contributed the most revenue?
7. Do VIP or Loyal customers spend more than New or Returning customers?
8. Which cities generated the highest number of transactions and revenue?
9. What payment methods were used most frequently?
10. What hours of the day had the highest transaction volume?
11. Are weekend transactions different from weekday transactions?
12. How does discount percentage relate to purchase amount?
13. Which age groups purchased the most during the promotional period?
14. Which products were purchased most frequently?
15. Are high-discount transactions associated with higher quantities sold?

### Bahasa Indonesia

1. Berapa jumlah transaksi selama periode observasi?
2. Berapa total revenue yang dihasilkan selama periode Black Friday sales?
3. Bagaimana perbandingan revenue Black Friday dengan hari non-Black Friday?
4. Kategori produk mana yang menghasilkan revenue tertinggi?
5. Kategori produk mana yang memiliki rata-rata diskon tertinggi?
6. Segment pelanggan mana yang paling banyak berkontribusi terhadap revenue?
7. Apakah pelanggan VIP atau Loyal memiliki nilai pembelian lebih tinggi dibanding New atau Returning?
8. Kota mana yang menghasilkan jumlah transaksi dan revenue tertinggi?
9. Metode pembayaran apa yang paling sering digunakan?
10. Jam berapa transaksi paling banyak terjadi?
11. Apakah transaksi akhir pekan berbeda dari transaksi hari kerja?
12. Bagaimana hubungan persentase diskon dengan nilai pembelian?
13. Kelompok usia mana yang paling banyak melakukan pembelian selama periode promosi?
14. Produk mana yang paling sering dibeli?
15. Apakah transaksi dengan diskon tinggi berkaitan dengan jumlah unit terjual yang lebih besar?

## Useful Derived Features

### English

You can create additional features from this dataset:

| Derived Feature | Formula / Idea | Purpose |
| --- | --- | --- |
| `datetime` | Combine `purchase_date` and `purchase_hour` | Create a complete transaction timestamp for time series analysis. |
| `day_name` | Extract day name from `purchase_date` | Compare sales by day of week. |
| `month` | Extract month from `purchase_date` | Useful if the dataset is expanded to multiple months. |
| `discount_amount` | `original_price - final_price` | Measures absolute discount value per unit. |
| `total_discount_amount` | `(original_price - final_price) * quantity` | Measures total discount value per transaction. |
| `revenue_before_discount` | `original_price * quantity` | Estimates transaction value before discount. |
| `effective_discount_pct` | `total_discount_amount / revenue_before_discount * 100` | Measures actual discount rate at transaction level. |
| `basket_size` | `quantity` | Represents number of items purchased in a transaction. |
| `avg_item_value` | `purchase_amount / quantity` | Measures average paid price per item. |
| `is_high_discount` | Flag transactions with discount above a selected threshold | Useful for comparing high-discount and low-discount transactions. |
| `customer_transaction_count` | Count transactions by `customer_id` | Identifies repeat or high-frequency customers. |
| `customer_total_spend` | Sum `purchase_amount` by `customer_id` | Identifies high-value customers. |

### Bahasa Indonesia

Kamu bisa membuat fitur tambahan dari dataset ini:

| Fitur Turunan | Formula / Ide | Tujuan |
| --- | --- | --- |
| `datetime` | Gabungkan `purchase_date` dan `purchase_hour` | Membuat timestamp transaksi lengkap untuk analisis time series. |
| `day_name` | Ekstrak nama hari dari `purchase_date` | Membandingkan penjualan berdasarkan hari. |
| `month` | Ekstrak bulan dari `purchase_date` | Berguna jika dataset diperluas ke beberapa bulan. |
| `discount_amount` | `original_price - final_price` | Mengukur nilai diskon absolut per unit. |
| `total_discount_amount` | `(original_price - final_price) * quantity` | Mengukur total nilai diskon per transaksi. |
| `revenue_before_discount` | `original_price * quantity` | Mengestimasi nilai transaksi sebelum diskon. |
| `effective_discount_pct` | `total_discount_amount / revenue_before_discount * 100` | Mengukur tingkat diskon aktual pada level transaksi. |
| `basket_size` | `quantity` | Merepresentasikan jumlah item dalam satu transaksi. |
| `avg_item_value` | `purchase_amount / quantity` | Mengukur rata-rata harga yang dibayar per item. |
| `is_high_discount` | Tandai transaksi dengan diskon di atas threshold tertentu | Berguna untuk membandingkan transaksi diskon tinggi dan rendah. |
| `customer_transaction_count` | Hitung jumlah transaksi per `customer_id` | Mengidentifikasi pelanggan repeat atau pelanggan dengan frekuensi tinggi. |
| `customer_total_spend` | Jumlahkan `purchase_amount` per `customer_id` | Mengidentifikasi pelanggan bernilai tinggi. |

## Important Notes

### English

- This dataset is transaction-level, so each row represents one purchase transaction, not one customer or one product summary.
- `purchase_amount` is the main revenue metric and is calculated from `final_price * quantity`.
- `final_price` is derived from `original_price` and `discount_pct`.
- `purchase_date` should be converted to datetime before time-based analysis.
- `purchase_hour` ranges from 0 to 23 and can be combined with `purchase_date` to create a complete timestamp.
- `is_black_friday` is the key variable for comparing Black Friday and non-Black Friday behavior.
- `is_weekend` can be treated as a boolean flag, even though it is stored as 0 and 1.
- The dataset covers a short time window from 2025-11-24 to 2025-12-01, so it is better for promotional-period analysis than long-term trend analysis.
- The distribution of some categorical variables appears relatively balanced, so the dataset may be synthetic or simulated. Insights should be framed as exploratory, not as proof of real retail behavior.
- Customer demographic variables such as `gender` and `age_group` should be used responsibly and should not be interpreted as causal factors without further evidence.

### Bahasa Indonesia

- Dataset ini berada pada level transaksi, sehingga setiap baris merepresentasikan satu transaksi pembelian, bukan ringkasan per pelanggan atau per produk.
- `purchase_amount` adalah metrik revenue utama dan dihitung dari `final_price * quantity`.
- `final_price` merupakan hasil turunan dari `original_price` dan `discount_pct`.
- `purchase_date` sebaiknya dikonversi menjadi datetime sebelum melakukan analisis berbasis waktu.
- `purchase_hour` memiliki rentang 0 sampai 23 dan dapat digabungkan dengan `purchase_date` untuk membuat timestamp lengkap.
- `is_black_friday` adalah variabel utama untuk membandingkan perilaku transaksi pada Black Friday dan non-Black Friday.
- `is_weekend` dapat diperlakukan sebagai flag boolean, meskipun disimpan dalam bentuk 0 dan 1.
- Dataset hanya mencakup periode pendek dari 2025-11-24 sampai 2025-12-01, sehingga lebih cocok untuk analisis periode promosi daripada analisis tren jangka panjang.
- Distribusi beberapa variabel kategorikal terlihat cukup seimbang, sehingga dataset ini kemungkinan bersifat sintetis atau simulasi. Insight sebaiknya diposisikan sebagai eksplorasi, bukan bukti perilaku retail aktual.
- Variabel demografi pelanggan seperti `gender` dan `age_group` perlu digunakan secara bertanggung jawab dan tidak boleh langsung dianggap sebagai faktor kausal tanpa bukti tambahan.