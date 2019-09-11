from asr import ASR

# ********* EXAMPLE SCENARIO ***************
S0 = 45 #------->EUR
sigma = 0.6 #--->EUR.day*-1/2 which corresponds to an annual volatility approximately equal to 21%.
T = 63 #-------->trading days 
N = [22, 62] #-->The set of possible dates for delivery before expiry
V = 4000000 #--->stocks day-1
Q = 20000000 #-->stocks
eta = 0.1 #----->EUR stock-1 day-1
fi = 0.75
gamma = 2.5e-7 #>EUR-1

NQ = 10 # the computational grid for q
INF = 1e9 # value for the infinity

scenario = ASR(S0, sigma, T, N, V, Q, eta, fi, gamma)
scenario.initialize(NQ, INF)

# uncomment 2 of the 3 following lines to calculate and save TETAs
# - use 'save_TETAs()' to save results to a text file
# - use 'save_gzip_TETAs()' to save results to a gzip file
scenario.get_TETAs()
#scenario.save_TETAs()
scenario.save_gzip_TETAs()

# uncoment the 2 of the 4 following lines to read TETAs from a file:
# - use 'read_TETAs' to read from a text file
# - use 'read_gzip_TETAs' to read from a gzip file
#filename = 'teta_qgrid_50_gamma_2.5e-07.txt' # define a filename to read TETAs
#scenario.read_TETAs(filename)
#filename = 'teta_qgrid_50_gamma_2.5e-07.gzip' # define a filename to read TETAs
#scenario.read_gzip_TETAs(filename)


# output price of the ASR contract
print('\n------------ OUTPUT ----------------')
scenario.get_PI()
print('Price of the ASR contract: PI/Q = {}'.format(scenario.PI))

# simulate price trajectory

# ucomment the following line to choose an example price trajectory
scenario.set_example_S(1) # specify the number of the example price trajectory: 1, 2 or 3

# ucomment the following line to generate a new price trajectory
# scenario.set_S()

# process a price trajectory
scenario.get_A()
scenario.get_Z()
scenario.get_q()

# save results
scenario.save_results()

# plot results
scenario.plot_trajectory()
scenario.plot_otimal_strategy()
