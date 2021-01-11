from collections.abc import Iterable
import itertools as it
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.dummy import DummyClassifier

#### Constants
PRICE = "close"
RETURN = "return"
TARGET_COL = f"{RETURN}_sign"
RANDOM_SEED = 42
SKLEARN_RANDOM_SEED = 42
###


########################################################################################
#           SKLEARN MODEL UTILITY FUNCTIONS
########################################################################################

def compare_dummy_clf(clf_score, X_in, y_in):
    cv_dummy_scores = cross_val_score(DummyClassifier(random_state=SKLEARN_RANDOM_SEED), X_in, y_in, cv=5, scoring="accuracy", n_jobs=-1)
    dummy_score = np.mean(cv_dummy_scores)
    
    if clf_score < dummy_score:
        print(f"Classifer, with a score of {clf_score:.5}, performs WORSE than a Dummy classifier with a score of {dummy_score:.5}")
    else:
        print(f"Classifer, with a score of {clf_score:.5}, performs BETTER than a Dummy classifier with a score of {dummy_score:.5}")
        
def get_confusion_matrix(clf, X_in, y_in, cv=5):
    y_pred_cv = cross_val_predict(clf, X_in, y_in.ravel(), cv=cv, n_jobs=-1)
    conf_mx = confusion_matrix(y_in.ravel(), y_pred_cv)
    return (conf_mx, y_pred_cv)

def plot_confusion_matrix(conf_mx, title=None):
    data = conf_mx[::-1]
    
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5])
    fig1 = ff.create_annotated_heatmap(data,
                                       x=["-ve", "+ve"], y=["+ve", "-ve"], colorscale=px.colors.sequential.Magma)
    
    fig2 = ff.create_annotated_heatmap(np.round(data / conf_mx.sum(axis=1), 4),
                                       x=["-ve", "+ve"], y=["+ve", "-ve"], colorscale=px.colors.sequential.Magma)
    
    annot1 = list(fig1.layout.annotations)
    annot2 = list(fig2.layout.annotations)
    
    for k  in range(len(annot2)):
        annot2[k]['xref'] = 'x2'
        annot2[k]['yref'] = 'y2'
    
    fig.add_trace(fig1.data[0], row=1, col=1)
    fig.add_trace(fig2.data[0], row=1, col=2)
    
    fig.update_layout(annotations=annot1+annot2, title=title)
    fig.update_xaxes(title_text="Predicted", row=1, col=1)
    fig.update_yaxes(title_text="Actual", row=1, col=1)
    
    fig.update_xaxes(title_text="Predicted", row=1, col=2)
    
    fig.update_layout(height=400, width=800)
    return fig


########################################################################################
#            FEATURE ENGINEERING UTILITY FUNCTIONS
########################################################################################

def feature_engineering(df_clean_dict, use_mom_ma=False):

    def add_lags(df, col, nlags):
        """
        Pass in a column name and lagged values will be added to dataframe
        """
        col_list = []

        if isinstance(nlags, Iterable):
            lags = nlags
        else:
            lags = range(1, nlags+1)

        for i in lags:
            lag = i
            col_name = f"{col}_t-{lag}"
            df[col_name] = df[col].shift(lag)

            col_list.append(col_name)

        return col_list

    def add_return(df, days=1):
        """
        Add returns for specified number of days.


        return: return column name
        """

        # days must be a minimum of 1
        days = max(1, days)

        if days == 1:
            return_col = RETURN

        else:
            return_col = f"{RETURN}{days}D"

        df[return_col] = np.log(df[PRICE]/df[PRICE].shift(days))

        return return_col
    

    def add_return_pct(df, days=1):
        """
        Add return percentage for specified number of days.


        return: return percentage column name
        """

        # days must be a minimum of 1
        days = max(1, days)

        if days == 1:
            return_col = f"{RETURN}_pct"

        else:
            return_col = f"{RETURN}{days}D_pct"

        df[return_col] = df[PRICE].pct_change(periods=days)

        return return_col
    
    def add_momentum(df, days=1):
        """
        Add momentum

        return: return column name
        """
        # days must be a minimum of 1
        days = max(1, days)

        mom_col = f"MOM{days}D"

        # Add log return
        df[mom_col] = df[PRICE] - df[PRICE].shift(days)

        return mom_col

    def add_sma(df, period=5, use_mom_ma=False):

        # Period can be a minimum of 5 for SMA
        period = max(5, period)

        sma_col = f"sma_{period}D"

        df[sma_col] = df[PRICE].rolling(window=period).mean()
        
        if use_mom_ma:
            df[sma_col] = df[sma_col] - df[sma_col].shift(1)
            
        return sma_col

    def add_ewm(df, period=7, use_mom_ma=False):

        # Period can be a minimum of 7 for EWMA
        period = max(7, period)

        ewm_col = f"EWM{period}D"

        df[ewm_col] = df[PRICE].ewm(span=period).mean()
        
        if use_mom_ma:
            df[ewm_col] = df[ewm_col] - df[ewm_col].shift(1)
               
        return ewm_col

    def add_std(df, period=7, use_mom_ma=False):

        # Period can be a minimum of 7 for std
        period = max(7, period)

        std_col = f"STD{period}D"

        # Calculate sample standard deviation
        df[std_col] = df[PRICE].rolling(window=period).std()
 
        if use_mom_ma:
            df[std_col] = df[std_col] - df[std_col].shift(1)
        
        return std_col

    #### User Settings

    # returns to calculate
    returns_ndays = [1, 5]

    # moomentum
    mom_ndays = [1, 5]

    # window for SMA
    sma_ndays = [7, 14, 21]      

    # ndays that ewm decays to
    ewm_ndays = [7, 14, 21]     

    # std window
    std_ndays = [7, 14, 21]      
    ############

    # Hold augmented feature set dataframes
    df_aug_dict = dict()

    # Hold column signs
    return_cols= []
    momentum_cols = []
    sma_cols, ewm_cols, std_cols = [],[], []


    for j, symbol, df in zip(it.count(0), df_clean_dict.keys(), df_clean_dict.values()):

        # Make a copy
        df_copy = df.copy()

        # RETURNS
        for i in returns_ndays:

            ret_col= add_return(df_copy, days=i)
            ret_lagged_cols = add_lags(df_copy, col=ret_col, nlags=7)

            # Add columns
            if j == 0:
                return_cols.append(ret_col)
                return_cols.extend(ret_lagged_cols)
        
        # RETURN PERCENTAGE
        ret_pct_col = add_return_pct(df_copy, days=1)
        if j == 0:
            return_cols.append(ret_pct_col)
            
        # MOMENTUM
        for i in mom_ndays:

            mom_col = add_momentum(df_copy, days=i)
            mom_lagged_cols = add_lags(df_copy, col=mom_col, nlags=7)

            if j == 0:
                momentum_cols.append(mom_col)
                momentum_cols.extend(mom_lagged_cols)

        # SMA - simple moving average
        for i in sma_ndays:
            sma_col = add_sma(df_copy, i, use_mom_ma=use_mom_ma)

            if j == 0:
                sma_cols.append(sma_col)

        # EWMA - exponential weighted moving average
        for i in ewm_ndays:
            ewm_col = add_ewm(df_copy, i, use_mom_ma=use_mom_ma)

            if j == 0:
                ewm_cols.append(ewm_col)

        # STD SMA - moving average standard deviation of sample
        for i in std_ndays:

            std_col = add_std(df_copy, i, use_mom_ma=use_mom_ma)

            if j == 0:
                std_cols.append(std_col)

        # Drop NaN values
        df_copy.dropna(how="any", inplace=True)

        # Add column for return sign (target label)
        df_copy[TARGET_COL] = np.sign(df_copy[RETURN]).astype(np.int)

        # Save to dict
        df_aug_dict[symbol] = df_copy

    output = dict()
    output["returns_cols"] = return_cols
    output["momentum_cols"] = momentum_cols
    output["sma_cols"] = sma_cols
    output["ewm_cols"] = ewm_cols
    output["std_cols"] = std_cols
    output["df_dict"] = df_aug_dict
    return output
