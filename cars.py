
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import os

# Task 0

def read_data(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print("The filename is invalid")
        return None

# Task 1

def task1(data):
    # 1(i)
    no_accident = data[data['Been in traffic incident'].str.lower() == 'no']
    st_dev = round(np.std(no_accident['Price'], ddof=1), 2)

    # 1(ii)
    sedans = data[data['Body'].str.lower() == 'sedan']
    mileage = sedans['Mileage'].values
    median_lower = np.median(mileage[mileage <= np.percentile(mileage, 25)])
    median_upper = np.median(mileage[mileage >= np.percentile(mileage, 75)])
    median_result = np.array([round(median_lower, 2), round(median_upper, 2)])

    # 1(iii)
    registered = data[(data['Registration'].str.lower() == 'yes') & (data['Year'] > 2010)]
    if len(registered) < 2:
        corr = 0
    else:
        vector = registered['Price'] * registered['EngineV']
        corr = round(np.corrcoef(vector, registered['Mileage'])[0, 1], 2)

    # 1(iv)
    cols = ['Price', 'Mileage', 'EngineV', 'Year', 'Fuel Economy']
    subset = data[cols].copy()
    for col in cols:
        if subset[col].dtype == 'object':
            subset[col] = pd.to_numeric(subset[col], errors='coerce')
    subset.fillna(0, inplace=True)
    standardized = (subset - subset.mean()) / subset.std()
    cov_matrix = np.cov(standardized.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    top_eigenvector = eigenvectors[:, 0]
    pca = np.dot(standardized.values, top_eigenvector.reshape(-1, 1)).round(2)

    return st_dev, median_result, corr, pca

# Task 2

def task2(data):
    # 2(i)
    no_accident = data[data['Been in traffic incident'].str.lower() == 'no']
    mean_price = no_accident['Price'].mean()
    filtered = no_accident[no_accident['Price'] < mean_price]
    task2i = filtered['Price'].round(2).values

    # 2(ii)
    task2ii = data.isna().any().any()
    if task2ii:
        data.fillna(0, inplace=True)

    return task2i, task2ii

# Task 3

def task3(data):
    # 3(i)
    x = sp.Symbol('x')
    expr = 1.5 * x**3 + 2 * x**2 + 5.5 * x + 7
    derivative = sp.diff(expr, x)

    petrol = data[data['Engine Type'].str.lower() == 'petrol']
    diesel = data[data['Engine Type'].str.lower() == 'diesel']
    f = sp.lambdify(x, derivative, 'numpy')
    petrol_derivatives = f(petrol['EngineV'].values)
    diesel_derivatives = f(diesel['EngineV'].values)
    task3i = [petrol_derivatives, diesel_derivatives]

    # 3(ii)
    Y, Z = sp.symbols('Y Z')
    X = Y**0.25 + Z**0.34
    diff_Y = sp.diff(X, Y)
    diff_Z = sp.diff(X, Z)
    f_Y = sp.lambdify((Y, Z), diff_Y, 'numpy')
    f_Z = sp.lambdify((Y, Z), diff_Z, 'numpy')

    engineV = data['EngineV'].values
    fuelEco = data['Fuel Economy'].values
    median_fuel = np.median(fuelEco)
    median_engine = np.median(engineV)

    result_Y = np.array(f_Y(engineV, median_fuel))
    result_Z = np.array(f_Z(median_engine, fuelEco))
    task3ii = result_Y - result_Z

    return task3i, task3ii

# Task 4

def task4(data):
    filtered = data[(data['Mileage'] <= 300) & (data['Registration'].str.lower() == 'yes')]
    if len(filtered) < 2:
        return 0
    return round(np.corrcoef(filtered['Mileage'], filtered['Price'])[0, 1], 2)

# Task 5

def task5(data, pca):
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.delaxes(axes[2, 1])

    # 5(i)
    filtered = data[(data['Mileage'] >= 100) & (data['Been in traffic incident'].str.lower() == 'no')]
    colors = ['red' if r.lower() == 'no' else 'green' for r in filtered['Registration']]
    axes[0, 0].bar(range(len(filtered)), filtered['Price'], color=colors)
    axes[0, 0].set_title('Mileage of cars separated by registration')

    # 5(ii)
    registered = data[data['Registration'].str.lower() == 'yes']
    body_price = registered.groupby('Body')['Price'].sum()
    axes[0, 1].pie(body_price, labels=body_price.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Price distribution for cars with registration, by body type')

    # 5(iii)
    engine_types = data['Engine Type'].unique()
    price_data = [data[data['Engine Type'] == et]['Price'] for et in engine_types]
    axes[1, 0].boxplot(price_data, labels=engine_types)
    axes[1, 0].set_title('Price Distribution by Engine Type')

    # 5(iv)
    filtered = data[(data['Mileage'] <= 400) & (data['Price'] >= 20000)]
    axes[1, 1].scatter(filtered['Mileage'], filtered['Price'])
    axes[1, 1].set_title('Scatter Plot of Price vs Mileage')
    axes[1, 1].set_xlabel('Mileage')
    axes[1, 1].set_ylabel('Price')

    # 5(v)
    filtered = data[data['Price'] <= 20000]
    pca_filtered = pca[data['Price'] <= 20000].flatten()
    range_ = pca_filtered.max() - pca_filtered.min()
    normed = (pca_filtered - pca_filtered.min()) / range_ if range_ != 0 else np.zeros_like(pca_filtered)
    colors = plt.cm.viridis(normed)
    axes[2, 0].bar(range(len(filtered)), filtered['Price'], color=colors)
    axes[2, 0].set_title('Car price coloured by PCA')

    plt.tight_layout()
    plt.show()

# Main

def main(fileName):
    data = read_data(fileName)
    if data is None:
        return (0, [0, 0], 0, np.array([]), np.array([]), False, [np.array([]), np.array([])], np.array([]), 0)

    if data.isna().any().any():
        data.fillna(0, inplace=True)

    st_dev, median_result, corr, pca = task1(data)
    task2i, task2ii = task2(data)
    task3i, task3ii = task3(data)
    task4i = task4(data)
    task5(data, pca)

    return (st_dev, median_result, corr, pca,
            task2i, task2ii,
            task3i, task3ii,
            task4i)

