import numpy as np
import glob
import pandas as pd
import multiprocessing as mp
import pickle

final_cluster = {0: ['TEL', 'PX', 'APA', 'NOV', 'ARNC', 'FRT', 'LUK', 'RCL', 'MRO', 'AXP', 'HRB', 'HSY', 'RJF', 'CSX', 'FMC', 'NI', 'SLB', 'BLL', 'ABC', 'CMI', 'FAST', 'MOS', 'HES', 'KMI', 'KMX', 'AME', 'BAX', 'WMB', 'MAR', 'IFF', 'PWR', 'COO', 'KEY', 'JBHT', 'RF', 'BF.B', 'FCX', 'AMG', 'HBAN', 'ZION', 'APD', 'HP', 'CVX', 'NFX', 'UNP', 'CB', 'BBT', 'BEN', 'HRS', 'APC', 'EXR', 'WYNN', 'PBCT', 'AWK', 'UTX', 'PG', 'COP', 'AJG', 'GPN', 'PXD', 'PFG', 'VTR', 'FIS', 'NRG', 'ABT', 'HST', 'BWA', 'TAP', 'CXO', 'DVN', 'CNC', 'MTD', 'AVGO', 'COG', 'MTB', 'CMA', 'UDR'],
 1: ['CTXS', 'CELG', 'UAA', 'TXN', 'ISRG', 'MCK', 'BSX', 'IDXX', 'CMS', 'WU', 'FFIV', 'XRX', 'EA', 'VRSN', 'JNPR', 'MHK', 'DLR', 'AKAM', 'RTN', 'NFLX', 'EXPE', 'GLW', 'FB', 'DGX', 'TSS', 'GM', 'NDAQ', 'EW', 'FLIR', 'DHI', 'CLX', 'WHR', 'PHM', 'GT', 'RHI', 'MA', 'CDNS', 'ROP', 'HOLX', 'TWX', 'MAS', 'GOOGL', 'ESRX', 'XYL', 'WYN', 'AET', 'V', 'EMN', 'ANTM', 'AEP', 'AAPL', 'CAH'],
 2: ['K', 'COST', 'NBL', 'SNPS', 'GGP', 'GPS', 'SNA', 'PSA', 'PCG', 'DVA', 'EFX', 'CERN', 'WMT', 'CHK', 'GWW', 'PRGO', 'LB', 'LNT', 'ADI', 'NWL', 'RL', 'HSIC', 'INCY', 'ESS', 'CHTR', 'JCI', 'OKE', 'KSS', 'VIAB', 'SLG', 'HCP', 'DISCA', 'TDG', 'NKE', 'TJX', 'AAP', 'WDC', 'AMD', 'SCG', 'TIF', 'KR', 'XRAY', 'M', 'AGN', 'FL', 'AYI', 'EQT', 'BDX', 'HBI', 'SJM', 'FTI', 'NCLH', 'PDCO', 'RRC', 'HII', 'CMG', 'MAT', 'IT', 'TRIP', 'RE', 'AZO', 'ALXN', 'CVS', 'CTL', 'CPB', 'DISCK', 'CBOE', 'MAC', 'HRL', 'VMC', 'SPG', 'MKC', 'MYL', 'JWN', 'ORLY', 'ARE', 'ANSS', 'NVDA', 'REG', 'TGT', 'KIM', 'CHD', 'VFC', 'SRCL', 'COL', 'FE', 'ALB', 'SRE'],
 3: ['PSX', 'ADM', 'TMK', 'FLR', 'AIV', 'PNR', 'GRMN', 'XL', 'LH', 'HAL', 'VRSK', 'KORS', 'EMR', 'MON', 'SYY', 'PH', 'DE', 'CBS', 'ADSK', 'IRM', 'ATVI', 'PGR', 'AMGN', 'MCHP', 'DG', 'PVH', 'DIS', 'TPR', 'SEE', 'ALL', 'HUM', 'AES', 'CHRW', 'MLM', 'PCLN', 'WM', 'DLTR', 'MSI', 'ABBV', 'ETR', 'CNP', 'PEG', 'CTAS', 'EOG', 'MNST', 'CMCSA', 'ED', 'YUM', 'GILD', 'LLY', 'SO', 'RSG', 'ECL', 'OXY', 'HCN', 'SWKS', 'AMAT', 'PRU', 'MDT', 'LYB', 'LLL', 'JEC', 'NEE', 'EL', 'MCO', 'CI', 'EXC', 'AAL', 'TSN', 'XOM', 'MCD', 'AEE', 'DUK', 'EXPD', 'XEC', 'L', 'WEC', 'FISV', 'PNW', 'PPG', 'EIX', 'ETN', 'AIZ'],
 4: ['FITB', 'BLK', 'STZ', 'LNC', 'XEL', 'MO', 'INTC', 'MSFT', 'ITW', 'MS', 'AVB', 'SCHW', 'CCL', 'LKQ', 'IP', 'PM', 'VNO', 'TMO', 'BA', 'UNM', 'WFC', 'AIG', 'PNC', 'PEP', 'CINF', 'MRK', 'VZ', 'JPM', 'NEM', 'PKI', 'FBHS', 'DFS', 'EQIX', 'NTAP', 'CF', 'ZTS', 'ANDV', 'GIS', 'PFE', 'ACN', 'AMP', 'ROST', 'SPGI', 'KO', 'ALK', 'RHT', 'ADBE', 'MDLZ', 'CME', 'LOW', 'GD', 'LEN', 'PAYX', 'TXT', 'ADS', 'DTE', 'VLO', 'DRI', 'JNJ', 'COF', 'A', 'GS', 'IVZ', 'INTU', 'USB', 'CRM', 'FDX', 'NOC', 'HD', 'STT', 'D', 'ULTA', 'MU', 'BK', 'CTSH', 'SBAC', 'PPL', 'WBA', 'AMT', 'BAC', 'CAG', 'AFL', 'MMC', 'DISH', 'HIG', 'BXP', 'C', 'PLD', 'MPC', 'MMM', 'ETFC', 'SYMC', 'UNH', 'EQR', 'AON', 'WY', 'TRV', 'STI', 'CSCO', 'BBY', 'LMT', 'MET', 'BRK.B', 'CBG', 'DPS', 'DAL'],
 5: ['SYK', 'NLSN', 'MGM', 'IR', 'NSC', 'TSCO', 'VAR', 'KLAC', 'KSU', 'CAT', 'AMZN', 'HCA', 'EBAY', 'T', 'AOS', 'ILMN', 'SNI', 'IPG', 'QCOM', 'AVY', 'OMC', 'F', 'URI', 'LUV', 'APH', 'BIIB', 'XLNX', 'DRE', 'ROK', 'LRCX', 'LEG', 'CCI', 'FLS', 'ALGN', 'ZBH', 'SBUX', 'WAT', 'DOV', 'CL', 'SHW', 'UPS', 'NTRS', 'ADP', 'PCAR', 'KMB', 'HAS', 'NUE', 'RMD', 'CA', 'IBM', 'GPC', 'UHS', 'PKG', 'TROW', 'HOG', 'SWK', 'STX', 'HON', 'UAL'],
 'Outlier': ['VRTX', 'REGN'],
 6: ['SIG', 'MAA'],
 7: ['GE']}

stock_data = []


def euclidean_dist(t1, t2):
    dist = 0
    for j in range(len(t1)):
        dist = dist + (t1[j] - t2[j]) ** 2
    return dist


def lb_keogh(s1, s2, r):
    lb_sum = 0
    for ind, i in enumerate(s1):

        lower_bound = np.amin(s2[(ind - r if ind - r >= 0 else 0):(ind + r)], axis=0)
        upper_bound = np.amax(s2[(ind - r if ind - r >= 0 else 0):(ind + r)], axis=0)

        for j in range(len(i)):
            if i[j] > upper_bound[j]:
                lb_sum = lb_sum + (i[j] - upper_bound[j]) ** 2
            elif i[j] < lower_bound[j]:
                lb_sum = lb_sum + (i[j] - lower_bound[j]) ** 2

    return np.sqrt(lb_sum)


def dtw_distance(s1, s2, w=None):
    """
    Calculates dynamic time warping Euclidean distance between two
    sequences. Option to enforce locality constraint for window w.
    """
    dtw = {}
    if w:
        w = max(w, abs(len(s1) - len(s2)))

        for i in range(-1, len(s1)):
            for j in range(-1, len(s2)):
                dtw[(i, j)] = float('inf')

    else:
        for i in range(len(s1)):
            dtw[(i, -1)] = float('inf')
        for i in range(len(s2)):
            dtw[(-1, i)] = float('inf')

    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        if w:
            for j in range(max(0, i - w), min(len(s2), i + w)):
                dist = euclidean_dist(s1[i], s2[j])
                dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[(i - 1, j - 1)])
        else:
            for j in range(len(s2)):
                dist = euclidean_dist(s1[i], s2[j])
                dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[(i - 1, j - 1)])

    return np.sqrt(dtw[len(s1) - 1, len(s2) - 1])


class Silhouette:
    def __init__(self, stock_data):
        self.assignments = {}
        self.stock_data = stock_data

    def calc_silhouette_multiproc_util(self, args):
        final_silhouette = {}
        print("Stock: " + args[0])
        sum_local = 0
        for stock_local in final_cluster[args[1]]:
            sum_local = sum_local + lb_keogh(self.stock_data[args[0]], self.stock_data[stock_local], 5)
        a = sum_local / final_cluster[args[1]].__sizeof__()
        sum_inter_cluster = 0
        count_inter_cluster = 0
        for j in range(0, 6):
            if j != args[1]:
                for stock_inter_cluster in final_cluster[j]:
                    sum_inter_cluster = sum_inter_cluster + lb_keogh(self.stock_data[args[0]],
                                                                         self.stock_data[stock_inter_cluster], 5)
                    count_inter_cluster += 1
        b = sum_inter_cluster / count_inter_cluster
        final_silhouette[args[0]] = ((b - a) / max(b, a))
        print(final_silhouette)
        return final_silhouette

    def calc_silhouette(self):
        self.assignments = {}
        args = []
        for i in range(0, 6):
            for stock_i in final_cluster[i]:
                args.append([stock_i, i])

        pool = mp.Pool(processes=mp.cpu_count())
        assignments = pool.map(self.calc_silhouette_multiproc_util, args)
        pool.close()
        pool.join()
        return assignments


if __name__ == '__main__':
    mp.freeze_support()

    files = glob.glob('./preprocessed_stocks/*')
    stock_data = {}
    count = 0

    for i, f in enumerate(files):
        # read data for each stock
        t_data = pd.read_csv(f)

        # extract data from columns
        oc = np.reshape(np.asarray(t_data['o/c'].tolist()), (-1, 1))
        volume = np.reshape(np.asarray(t_data['volume'].tolist()), (-1, 1))
        high = np.reshape(np.asarray(t_data['high'].tolist()), (-1, 1))
        low = np.reshape(np.asarray(t_data['low'].tolist()), (-1, 1))

        # get stock ticker name and create 2D numpy array for values
        ticker = t_data['name'].tolist()[0]
        ts_data = np.hstack((oc, volume, high, low))

        # handle missing data by ignoring
        if ts_data.shape[0] == 1259:
            stock_data[ticker] = ts_data
    print(stock_data.keys())
    silhouette = Silhouette(stock_data)
    silhouette_score = silhouette.calc_silhouette()
    log_file = open('silhouette_log', 'wb')
    pickle.dump(silhouette_score, log_file)
    log_file.close()
    print(silhouette_score)


