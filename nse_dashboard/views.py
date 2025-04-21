# from django.shortcuts import render
# import nselib  # Base import
# import pandas as pd
# import logging

# logger = logging.getLogger(__name__)

# def dashboard_view(request):
#     context = {}
#     try:
#         # Access all functions through the main module
#         context.update({
#             'holidays': nselib.trading_holiday_calendar(),
#             'nifty50': nselib.capital_market.nifty50_equity_list(),
#             'market_indices': nselib.capital_market.market_watch_all_indices(),
#             'fii_dii': nselib.capital_market.fii_dii_trading_activity(),
#             'option_chain': nselib.derivatives.nse_live_option_chain("NIFTY"),
#         })
        
#         # Convert DataFrames to HTML
#         for key in context:
#             if isinstance(context[key], pd.DataFrame):
#                 context[key] = context[key].to_html(classes='table')
                
#     except Exception as e:
#         logger.error(f"Dashboard error: {str(e)}")
#         context['error'] = "Data temporarily unavailable"
    
#     return render(request, 'nse_dashboard/dashboard.html', context)
# nse_dashboard/views.py
# from django.shortcuts import render
# from nselib import capital_market  # Direct submodule import
# from nselib import derivatives
# import pandas as pd
# from nselib.capital_market import nifty50_equity_list

# def dashboard_view(request):
#     try:
#         context = {
#             'nifty50': capital_market.nifty50_equity_list(),
#             'option_chain': derivatives.nse_live_option_chain("NIFTY")
#         }
        
#         # Convert DataFrames to HTML
#         for key in context:
#             if isinstance(context[key], pd.DataFrame):
#                 context[key] = context[key].to_html(classes='table')
                
#     except Exception as e:
#         context = {'error': str(e)}
    
#     return render(request, 'nse_dashboard/dashboard.html', context)
# nse_dashboard/views.py
# nse_dashboard/views.py
# nse_dashboard/views.py
from django.shortcuts import render
import nselib
from nselib import trading_holiday_calendar


def dashboard_view(request):
   
    
    return render(request, 'nse_dashboard/dashboard.html')