import sys
from ts import data
from ts import model

### INPUT

# read input arguments
arguments = sys.argv

indicator = []
indicator_name = arguments[1]
indicator = data.getIndicator(indicator_name)

#model.plot("AR", indicator, model.AR(indicator))
validation = model.ARMA_validation_fc(indicator, 24)
#validation = model.AR_predict_N(indicator,120)
print validation
#model.plot_pred("AR val", indicator, model.AR(indicator), validation[0], validation[1])
model.plot_pred("AR val", indicator, model.ARMA(indicator), validation[0], validation[1])
print model.validation_prev(model.AR(indicator), 24)
