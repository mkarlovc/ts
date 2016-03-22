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
#validation = model.ARMA_val(indicator, 24)
#validation = model.validation_prev(indicator, 24)
#print validation[0]
#print validation[1]
#print validation[2]
#print validation
#model.plot_pred("AR val", indicator, model.AR(indicator), validation[0], validation[1])
#model.plot_pred("AR val", indicator, model.ARMA(indicator), validation[0], validation[1])
#print model.validation_prev(model.AR(indicator), 24)


f = model.feature_set(indicator, 10, 4, 24)
val = model.RF_fc(f[0], f[1])
model.plot_pred1("RF val", indicator, val[0], val[1], val[2]["R2"])
#print val
