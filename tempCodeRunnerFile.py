if 'Unnamed: 0' in heart_data.columns:
    heart_data = heart_data.drop(['Unnamed: 0'], axis=1)