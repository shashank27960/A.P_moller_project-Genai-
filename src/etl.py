import pandas as pd
import duckdb
import os

def load_csvs(data_dir='data/archive/'):
    orders = pd.read_csv(os.path.join(data_dir, 'olist_orders_dataset.csv'),
                         parse_dates=['order_purchase_timestamp', 'order_delivered_customer_date'])
    items = pd.read_csv(os.path.join(data_dir, 'olist_order_items_dataset.csv'))
    products = pd.read_csv(os.path.join(data_dir, 'olist_products_dataset.csv'))
    customers = pd.read_csv(os.path.join(data_dir, 'olist_customers_dataset.csv'))
    cat_trans = pd.read_csv(os.path.join(data_dir, 'product_category_name_translation.csv'))
    return orders, items, products, customers, cat_trans


def build_merged(orders, items, products, customers, cat_trans):
    df = (orders
          .merge(items, on='order_id', how='left')
          .merge(products, on='product_id', how='left')
          .merge(customers, on='customer_id', how='left'))

    # translate categories
    df = df.merge(cat_trans, on='product_category_name', how='left')
    df['category_en'] = df.get('product_category_name_english').fillna('unknown')
    return df


if __name__ == '__main__':
    orders, items, products, customers, cat_trans = load_csvs()
    df = build_merged(orders, items, products, customers, cat_trans)
    print(f"Merged dataframe shape: {df.shape}")
    df.head().to_csv('data/merged_preview.csv', index=False)
