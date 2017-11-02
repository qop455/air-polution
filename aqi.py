from numpy import isnan, nan

def pm25_to_aqi(pm25):
    if isnan(pm25):
        return nan
    else:
        pm25 = round(pm25, 1)
        if 0 <= pm25 <= 15.4:
            # 0 ~ 50
            return mapping(pm25, 0.0, 15.4, 0.0, 50.0)

        elif 15.5 <= pm25 <= 35.4:
            # 51 ~ 100
            return mapping(pm25, 15.5, 35.4, 51.0, 100.0)

        elif 35.5 <= pm25 <= 54.4:
            # 101 ~ 150
            return mapping(pm25, 35.5, 54.4, 101.0, 150.0)

        elif 54.5 <= pm25 <= 150.4:
            # 151 ~ 200
            return mapping(pm25, 54.5, 150.4, 151.0, 200.0)

        elif 150.5 <= pm25 <= 250.4:
            # 201 ~ 300
            return mapping(pm25, 150.5, 250.4, 201.0, 300.0)

        elif 250.5 <= pm25 <= 350.4:
            # 301 ~ 400
            return mapping(pm25, 250.5, 350.4, 301.0, 400.0)

        elif 350.5 <= pm25 <= 500.4:
            # 401 ~ 500
            return mapping(pm25, 350.5, 500.4, 401.0, 500.0)

        elif 500.5 <= pm25:
            # 500 ~ 
            return pm25 

        else:
            return nan

def mapping(value, xMin, xMax, yMin, yMax):
    # xMin ~ xMax >> yMin ~ yMax
    xSpan = xMax - xMin
    ySpan = yMax - yMin

    valueScaled = float(value - xMin) / xSpan
    valueMapped = yMin + (valueScaled * ySpan)
    
    return round(valueMapped)
