from sklearn.preprocessing import MinMaxScaler, StandardScaler


def standardScaler(trnX, tstX, trnY, tstY):

    scaler = StandardScaler()
    trnX = scaler.fit_transform(trnX)
    tstX = scaler.transform(tstX)

    return trnX, tstX, trnY, tstY


def minMaxScaler(trnX, tstX, trnY, tstY):

    scaler = MinMaxScaler()
    trnX = scaler.fit_transform(trnX)
    tstX = scaler.transform(tstX)

    return trnX, tstX, trnY, tstY
