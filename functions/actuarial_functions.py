import numpy as np
import pandas as pd


def get_termlife_annuity(age, duration, interest, A= 0.00022, B=2.7*10**(-6), c=1.124, shock_mort = 0):
    
    '''Calculate annuity of term life insurance'''

    v = 1/(1+interest)
    ann = 0
    for k in range(duration):
        ann+= v**k*np.exp(-A*k-B/np.log(c)*c**(age)*(c**k-1))*(1+shock_mort)                  
    
    return ann

 
def get_termlife_APV_ben(age, duration, interest, A= 0.00022, B=2.7*10**(-6), c=1.124, shock_mort = 0):

    '''Calculate apv of benefits for term life insurance'''
    
    v = 1/(1+interest)
    apv = 0
    for k in range(duration):
        apv += v**(k+1)*np.exp(-A*k-B/np.log(c)*c**(age)*(c**k-1))*(1+shock_mort)*(1-np.exp(-A-B/np.log(c)*c**(age+k)*(c-1)*(1+shock_mort)))
    
    return apv



def get_termlife_premium(age_init,Sum_ins,duration,  interest,  A= 0.00022, B=2.7*10**(-6), c=1.124):

    '''Calculate premium for term life insurance'''
    
    return Sum_ins*get_termlife_APV_ben(age_init, duration, interest, A, B, c)/get_termlife_annuity(age_init, duration, interest, A, B, c)



def get_termlife_reserve(age_curr, Sum_ins, duration,  interest, age_of_contract=0, A= 0.00022, B=2.7*10**(-6), c=1.124):

    '''Calculate current Policy value of term life insurance contract'''
    age_init = age_curr-age_of_contract
    prem = get_termlife_premium(age_init,Sum_ins, duration, interest, A,B,c)
    apv_prem = get_termlife_annuity(age_curr, duration-age_of_contract, interest, A,B,c)
    apv_ben = get_termlife_APV_ben(age_curr, duration-age_of_contract, interest, A,B,c)
    
    return Sum_ins*apv_ben -prem*apv_prem 



def get_termlife_reserve_profile(age_curr, Sum_ins, duration, interest,age_of_contract = 0,  A= 0.00022, 
                                 B=2.7*10**(-6), c=1.124, option_past = True, age_limit = 120, shock_mort=0, shock_int = 0):
    '''
    Calculate Policy values up to maturity of term life insurance contract
    Potentially even including the past policy values, starting at the start of the contract
    '''

    age_init = age_curr-age_of_contract
    premium = get_termlife_premium(age_init, Sum_ins,duration, interest, A,B,c) # not affected by potential shocks to interest rate or mortality!!
    reserve = np.zeros(duration+1)
    # No expenses
    e_ann = 0
    e_init = 0
    # No claims-related expenses
    E_claim = 0
    reserve[0] = - e_init
    reserve[-1] = 0
    for k in range(duration-1): # Exclude value of reserve at maturity of contract (set 0 by init. reserve, to avoid inaccuracy due to rounding errors)
        if (age_init+k < age_limit):
            # survival prob at age x+k
            prob_live = np.exp(-A-B/np.log(c)*c**(age_init+k)*(c-1)) # note: shock affects the whole surv. curve, i.e. each 1-year prob
            reserve[k+1] = ((reserve[k]+premium-e_ann)*(1+interest+shock_int) - (Sum_ins+E_claim)*(1-prob_live))/prob_live/(1+shock_mort)
        else:
            # switch to equivalence principle, as less prone to accumulate rounding errors for high ages/ low survival probs
            # note: dead if-case, as data are constructed w.r.t. age_limit
            reserve[k+1] = (Sum_ins+(e_ann+e_init))*get_termlife_APV_ben(age_init+k,duration,interest+shock_int, shock_mort=shock_mort)-\
                                premium*get_termlife_annuity(age_init+k, duration, interest+shock_int, shock_mort=shock_mort)
    
    if option_past==False:
        if age_of_contract <= duration:
            reserve = reserve[age_of_contract:]
        else:
            raise ValueError('Error in get_termlife_reserve_profile()!')

    return reserve   



def get_pension_reserve(fund_accum = 0, age = 40, salary = 1, salary_scale = 0.02, contribution = 0.03, 
                         A = 0.00022, B = 2.7*10**(-6), c = 1.124, interest = 0.01,
                        pension_age_max = 67,
                        early_pension = [0.1, 0.05, 0.05, 0.05, 0.05, 0.05]):

    '''
    Inputs: Accumulated Fund, Current Salary, Salary scale/ factor, Contribution Rate, SUS-Model (A,B,c)
         Rechnungszins (const.), max. Rentenalter, Frührente (%-Anteile für Alter 60-66)
         max. Alter not included as we assume fund to be paid as lump sum at pension age. 
         Annuitization of a lump sum can be considered as a seperate contract
    Output: Expected Reserve up to max age
    '''
        
    reserve = np.zeros(pension_age_max-age+1)
    reserve[0] = fund_accum
    for i in range(pension_age_max-age-len(early_pension)-1):
        reserve[i+1] = (reserve[i]+contribution*salary*(1+salary_scale)**(i))*(1+interest)*np.exp(-A-B/np.log(c)*c**(age+i)*(c-1))
    k=0
    for i in range(pension_age_max-age-len(early_pension)-1,pension_age_max-age-1):    
        # Old Reserve + Contribution - Mid-year exit in case of death 
        # or exit at beginning of year for early retirement
        reserve[i+1] = (reserve[i]+contribution*salary*(1+salary_scale)**(i))*(1+interest)*np.exp(-A-B/np.log(c)*c**(age+i)*(c-1))*(1-early_pension[k])     
        k+=1
    
    return reserve



def pension_reserve(data, pension_age_max = 67, age_min = 25, interest_std = 0.03, 

                    ep_structure = [0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]):

    '''
    Inputs:
    -------
        'Fund','Age', 'Salary', 'Salary_scale', 'Contribution', 'interest_rate'

    Outputs:
    --------
        Expected pension values of to retirement

    Note: matrix Version of get_pension_reserve
    '''
    
    length = pension_age_max - age_min + 1
    reserves = np.zeros([data.shape[0],length])
    
    if data.shape[1] == 6:
        for i in range(data.shape[0]):

            reserves[i,0:length - data[i,1].astype('int')+age_min] = get_pension_reserve( 
                                                                     age =data[i,1].astype('int'),
                                                                     fund_accum = data[i,0], 
                                                                     salary = data[i,2], salary_scale = data[i,3], 
                                                                     contribution = data[i,4], interest = data[i,5],
                                                                     pension_age_max = 67,
                                                                     early_pension = ep_structure)
    elif data.shape[1] == 5:
        
        for i in range(data.shape[0]):

            reserves[i,0:length - data[i,1].astype('int')+age_min] = get_pension_reserve(
                                                             age =data[i,1].astype('int'), 
                                                             fund_accum = data[i,0], 
                                                             salary = data[i,2], salary_scale = data[i,3], 
                                                             contribution = data[i,4], interest = interest_std,
                                                             pension_age_max = 67,
                                                             early_pension = ep_structure)
    
    
    return reserves



def get_historic_interest(aoc):

    '''
    Compute guaranteed interest rate of a contract based on historic 'Höchstrechnungszins' by DAV.
    See https://aktuar.de/unsere-themen/lebensversicherung/hoechstrechnungszins/Seiten/default.aspx

    Inputs:
    --------
        aoc: numpy array with values; number of years since contract was initiated.

    Outputs:
    ---------
        Guaranteed interest rate for all contracts which age of contract was provided for.

    '''
    interest = (aoc<=3)*0.009 + ((aoc>3)&(aoc<=5))*0.0125 + ((aoc>5)&(aoc<=8))*0.0175 + ((aoc>8)&(aoc<=13))*0.0225 + \
                ((aoc>13)&(aoc<=16))*0.0275 + ((aoc>16)&(aoc<=20))*0.0325 + (aoc>20)*0.04

    return interest



def target_investment_return( contracts, assets, counts = None, premiums = None, A= 0.00022, B=2.7*10**(-6), c=1.124):
    '''
    Compute the zero loss return the IC has to generate, i.e. a investment-value weighted return.
    '''

    if type(premiums) == type(None):
        premiums = np.zeros((len(contracts),))
        for i in range(len(contracts)):
            # contracts: 0: current age (!), 1: sum_ins, 2: duration, 3: elapsed duration, 4: interest rate
            premiums[i] = get_termlife_premium(age_init = contracts[i,0]-contracts[i,3], Sum_ins = contracts[i,1], duration = contracts[i,2].astype('int'),  interest = contracts[i,4], A= A, B=B, c=c)

    volume = premiums+assets
    return sum(contracts[:,-1]*volume/volume.sum())



def SCR_analysis(val_true, val_nn = (None, None), val_km = (None, None), A= 0.00022, B=2.7*10**(-6), c=1.124 ):

    '''
    Create table comparing SCR of true portfolio and grouping, in the light of the baseline szenario and mortality and interest rate shocks.

    Inputs:
    -------
        val_true:     tuple: contract data (term life only), i.e. age, sum_ins, duration, elapsed duration and interest rate, and fund volume (tuple: time 0 and 1)
        val_nn:       triplet: contract data of nn grouping, fund volume (tuple: time 0 and 1), member_counts
        val_km:       triplet: contract data of km grouping, fund volume (tuple: time 0 and 1), member_counts 
        *args         mortality parameters A, B, c
    Outputs:
    --------
    '''

    # unzip
    mp_true = val_true[0]
    mp_nn = val_nn[0]
    mp_km = val_km[0]
    asset_true = val_true[1]
    asset_nn = val_nn[1]
    asset_km = val_km[1]
    count_nn = val_nn[2]
    count_km = val_nn[2]

    # portfolio sizes
    N_true = len(val_true[0])
    # retrieve fair premiums
    premiums_true = np.zeros((N_true,))
    for i in range(N_true):
        # contracts: 0: current age (!), 1: sum_ins, 2: duration, 3: elapsed duration, 4: interest rate
        premiums_true[i] = get_termlife_premium(age_init = mp_true[i,0]-mp_true[i,3], Sum_ins = mp_true[i,1], 
                                                duration = mp_true[i,2].astype('int'),  interest = mp_true[i,4], A= A, B=B, c=c)

    # avg. interest in portfolio
    interest_avg = target_investment_return(contracts=mp_true, assets=asset_true[0], counts = None, premiums = premiums_true,A= A, B=B, c=c)


    # shocks on (interest rate/ mortality)
    # interest rate shock: additive, i.e. i' = i + shock
    # mortality shock: multiplicative on death prob, i.e. q' = q*+shock
    base = (interest_avg,0)
    shock1 = (interest_avg-0.01,0)
    shock2 = (interest_avg,0.01)
    shock3 = (interest_avg-0.006, 0.006)

    # Premiums NN
    if type(val_nn[0]) != type(None):
        K_nn = len(val_nn[0])
        premiums_nn = np.zeros((K_nn,))
        for i in range(K_nn):
            # note: respect integer-valued remaining duration and dictionary type of val_nn (!)
            nn_up = get_termlife_premium(age_init = mp_nn[i,0]-mp_nn[i,3], Sum_ins = mp_nn[i,1], 
                                                duration = np.ceil(mp_nn[i,2]).astype('int'),  interest = mp_nn[i,4], A= A, B=B, c=c)
            nn_low = get_termlife_premium(age_init = mp_nn[i,0]-mp_nn[i,3], Sum_ins = mp_nn[i,1], 
                                                duration = np.floor(mp_nn[i,2]).astype('int'),  interest = mp_nn[i,4], A= A, B=B, c=c)
            premiums_nn[i] = (nn_up+nn_low)/2

    # Premiums KM
    if type(val_km[0]) != type(None):
        K_km = len(val_km[0])
        if (K_km != K_nn):
            print('ValError: K_km != K_nn.')
            exit()
        premiums_km = np.zeros((K_km,))
        for i in range(K_km):
            # note: respect integer-valued remaining duration and dictionary type of val_nn (!)
            km_up = get_termlife_premium(age_init = (mp_km[i,0]-mp_km[i,3]), Sum_ins = mp_km[i,1], 
                                                duration = np.ceil(mp_km[i,2]).astype('int'),  interest = mp_km[i,4], A= A, B=B, c=c)
            km_low = get_termlife_premium(age_init = np.ceil(mp_km[i,0]-mp_km[i,3]), Sum_ins = mp_km[i,1], 
                                                duration = np.floor(mp_km[i,2]).astype('int'),  interest = mp_km[i,4], A= A, B=B, c=c)
            premiums_km[i] = (km_up+km_low)/2


    print('Overall premium income (True/ NN/ KM): ', round(premiums_true.sum()), '/ ', round((premiums_nn*count_nn).sum()), ' / ', round((premiums_km*count_km).sum()))
    print('Fund volume (true/ NN/ KM): ', (round(asset_true[0].sum()), round(asset_true[1].sum())), ' / ', 
            (round((asset_nn[0]*count_nn).sum()), round((asset_nn[1]*count_nn).sum())), ' / ', (round((asset_km[0]*count_km).sum()), round((asset_km[1]*count_km).sum())))
    print('Agg. sum insured (true/ NN/ KM): ', round(mp_true[:,1].sum()), ' / ', round((mp_nn[:,1]*count_nn).sum()), ' / ', round((mp_km[:,1]*count_km).sum()), '\n')

    # baseline
    scr_true_base = (one_step_investment_projection(contracts = mp_true, assets= asset_true, premiums = premiums_true, shock_int = base[0],  
                                        A= A, B= B, c= c),
                    one_step_risk_projection(contracts = mp_true, assets= asset_true, premiums = premiums_true, shock_mort = base[1], 
                                        A= A, B= B, c= c)
                    )

                
    # interest rate shock
    scr_true_int = (one_step_investment_projection(contracts = mp_true, assets= asset_true, premiums = premiums_true, shock_int = shock1[0],  
                                        A= A, B= B, c= c),
                    one_step_risk_projection(contracts = mp_true, assets= asset_true, premiums = premiums_true, shock_mort = shock1[1], 
                                        A= A, B= B, c= c)
                    )
    # mortality shock
    scr_true_mort = (one_step_investment_projection(contracts = mp_true, assets= asset_true, premiums = premiums_true, shock_int = shock2[0],  
                                        A= A, B= B, c= c),
                    one_step_risk_projection(contracts = mp_true, assets= asset_true, premiums = premiums_true, shock_mort = shock2[1], 
                                        A= A, B= B, c= c)
                    )
    # combined shock
    scr_true_comb = (one_step_investment_projection(contracts = mp_true, assets= asset_true, premiums = premiums_true, shock_int = shock3[0],  
                                        A= A, B= B, c= c),
                    one_step_risk_projection(contracts = mp_true, assets= asset_true, premiums = premiums_true, shock_mort = shock3[1], 
                                        A= A, B= B, c= c)
                    )

    ## Neural network SCR
    if type(val_nn[0]) != type(None):
        # baseline
        scr_nn_base = (one_step_investment_projection(contracts = mp_nn, assets= asset_nn, premiums = premiums_nn, counts= count_nn,
                                                        shock_int = base[0],  A= A, B= B, c= c),
                        one_step_risk_projection(contracts = mp_nn, assets= asset_nn, premiums = premiums_nn, counts= count_nn,
                                                         shock_mort = base[1], A= A, B= B, c= c)
                        )
        # interest rate shock
        scr_nn_int = (one_step_investment_projection(contracts = mp_nn, assets= asset_nn, premiums = premiums_nn, counts= count_nn,
                                                        shock_int = shock1[0],  A= A, B= B, c= c),
                        one_step_risk_projection(contracts = mp_nn, assets= asset_nn, premiums = premiums_nn, counts= count_nn,
                                                         shock_mort = shock1[1], A= A, B= B, c= c)
                        )
        # mortality shock
        scr_nn_mort = (one_step_investment_projection(contracts = mp_nn, assets= asset_nn, premiums = premiums_nn, counts= count_nn,
                                                        shock_int = shock2[0],  A= A, B= B, c= c),
                        one_step_risk_projection(contracts = mp_nn, assets= asset_nn, premiums = premiums_nn, counts= count_nn,
                                                         shock_mort = shock2[1], A= A, B= B, c= c)
                        )
        # interest rate shock
        scr_nn_comb = (one_step_investment_projection(contracts = mp_nn, assets= asset_nn, premiums = premiums_nn, counts= count_nn,
                                                        shock_int = shock3[0],  A= A, B= B, c= c),
                        one_step_risk_projection(contracts = mp_nn, assets= asset_nn, premiums = premiums_nn, counts= count_nn,
                                                         shock_mort = shock3[1], A= A, B= B, c= c)
                        )
    else:
        scr_nn_base, scr_nn_int, scr_nn_mort, scr_nn_comb = None, None, None, None


    ## KMeans SCR
    if type(val_nn[0]) != type(None):
        # baseline
        scr_km_base = (one_step_investment_projection(contracts = mp_km, assets= asset_km, premiums = premiums_km, counts= count_km,
                                                        shock_int = base[0],  A= A, B= B, c= c),
                        one_step_risk_projection(contracts = mp_km, assets= asset_km, premiums = premiums_km, counts= count_km,
                                                        shock_mort = base[1], A= A, B= B, c= c)
                        )
        # interest rate shock
        scr_km_int = (one_step_investment_projection(contracts = mp_km, assets= asset_km, premiums = premiums_km, counts= count_km,
                                                        shock_int = base[0], A= A, B= B, c= c),
                        one_step_risk_projection(contracts = mp_km, assets= asset_km, premiums = premiums_km, counts= count_km,
                                                         shock_mort = shock1[1], A= A, B= B, c= c)
                        )
        # mortality shock
        scr_km_mort = (one_step_investment_projection(contracts = mp_km, assets= asset_km, premiums = premiums_km, counts= count_km,
                                                        shock_int = base[0], A= A, B= B, c= c),
                        one_step_risk_projection(contracts = mp_km, assets= asset_km, premiums = premiums_km, counts= count_km,
                                                         shock_mort = shock2[1], A= A, B= B, c= c)
                        )
        # interest rate shock
        scr_km_comb = (one_step_investment_projection(contracts = mp_km, assets= asset_km, premiums = premiums_km, counts= count_km,
                                                        shock_int = base[0], A= A, B= B, c= c),
                        one_step_risk_projection(contracts = mp_km, assets= asset_km, premiums = premiums_km, counts= count_km,
                                                         shock_mort = shock3[1], A= A, B= B, c= c)
                        )
    else:
        scr_km_base, scr_km_int, scr_km_mort, scr_km_comb = None, None, None, None


    df = pd.DataFrame(data = None, index = ['P', 'P^(NN)', 'P^(KM)'] )
    df['base'] = [scr_true_base, scr_nn_base, scr_km_base]
    df['shock 1'] = [scr_true_int, scr_nn_int, scr_km_int]
    df['shock 2'] = [scr_true_mort, scr_nn_mort, scr_km_mort]
    df['shock 3'] = [scr_true_comb, scr_nn_comb, scr_km_comb]
    return df


def one_step_investment_projection(contracts, assets, counts = None, premiums = None, shock_int = 0, A= 0.00022, B=2.7*10**(-6), c=1.124):

    '''
    Come single period projection of investment surplus (current_reserve + premium_income)*(change_in_interest_rate), including shock szenarios, for SCR purpose.

    Inputs:
    -------
        contracts:  contracts data (term life only), i.e. age, sum_ins, duration, elapsed duration and interest rate
        assets:     tuple: policy value per contract at times 0 and 1
        shock_int:  additive interest rate shock; shock on assets
        shock_mort: multiplicative shock on mortality curve; shock on liabilites
        bool_separate: show investment and risk surplus separately

    Outputs:
    --------
        tuple of 1-period-projected assets and liabilities
    '''

    # retrieve fair premiums, based on original assumptions at underwriting of contracts (if not provided as input yet)
    if type(premiums) == type(None):
        premiums = np.zeros((len(contracts),))
        for i in range(len(contracts)):
            # contracts: 0: current age (!), 1: sum_ins, 2: duration, 3: elapsed duration, 4: interest rate
            premiums[i] = get_termlife_premium(age_init = contracts[i,0]-contracts[i,3], Sum_ins = contracts[i,1], duration = contracts[i,2].astype('int'),  interest = contracts[i,4], A= A, B=B, c=c)

    # key quantities
    interest_raw = contracts[:,4]
    interest_shocked = shock_int

    # compute assets to cover losses (only affected by interest shock, not mortality)
    inv_surplus = (assets[0]+premiums)*(interest_shocked-interest_raw)

    if type(counts) != type(None):
        inv_surplus *= counts

    return np.round_(inv_surplus.sum(),2)


def one_step_risk_projection(contracts, assets, counts = None, premiums = None, shock_mort = 0, A= 0.00022, B=2.7*10**(-6), c=1.124):

    '''
    Come single period projection of risk surplus (Sum_ins-reserve_next_year)*(change in mortality), including shock szenarios, for SCR purpose.

    Inputs:
    -------
        contracts:  contracts data (term life only), i.e. age, sum_ins, duration, elapsed duration and interest rate
        assets:     tuple: policy value per contract at times 0 and 1
        shock_int:  additive interest rate shock; shock on assets
        shock_mort: multiplicative shock on mortality curve; shock on liabilites
        bool_separate: show investment and risk surplus separately

    Outputs:
    --------
        tuple of 1-period-projected assets and liabilities
    '''
    # retrieve fair premiums, based on original assumptions at underwriting of contracts (if not provided as input yet)
    if type(premiums) == type(None):
        premiums = np.zeros((len(contracts),))
        for i in range(len(contracts)):
            # contracts: 0: current age (!), 1: sum_ins, 2: duration, 3: elapsed duration, 4: interest rate
            premiums[i] = get_termlife_premium(age_init = contracts[i,0]-contracts[i,3], Sum_ins = contracts[i,1], duration = contracts[i,2].astype('int'),  interest = contracts[i,4], A= A, B=B, c=c)

    # key quantities
    time = 1*((contracts[:,2]-contracts[:,3])>= 1) + (contracts[:,2]-contracts[:,3])*((contracts[:,2]-contracts[:,3])<1)
    prob_death_raw = (1-np.exp(-A*time-B/np.log(c)*c**(contracts[:,0])*(c**time-1))) # 1-year death prob for current age
    prob_death_shocked = prob_death_raw+shock_mort

    # compute expected death-claims
    risk_surplus = (contracts[:,1]-assets[1])*(prob_death_raw-prob_death_shocked)

    if type(counts) != type(None):
        risk_surplus *= counts
    return np.round_(risk_surplus.sum(),2)


def SCR_interest_analysis(val_true, val_nn = (None, None), val_km = (None, None), A= 0.00022, B=2.7*10**(-6), c=1.124 ):

    '''
    Create table looking at investment surplus of true portfolio and grouping, in the light of the baseline szenario and interest rate shocks.

    Inputs:
    -------
        val_true:     tuple: contract data (term life only), i.e. age, sum_ins, duration, elapsed duration and interest rate, and fund volume (tuple: time 0 and 1)
        val_nn:       triplet: contract data of nn grouping, fund volume (tuple: time 0 and 1), member_counts
        val_km:       triplet: contract data of km grouping, fund volume (tuple: time 0 and 1), member_counts 
        *args         mortality parameters A, B, c
    Outputs:
    --------
    '''

    # unzip
    mp_true = val_true[0]
    mp_nn = val_nn[0]
    mp_km = val_km[0]
    asset_true = val_true[1]
    asset_nn = val_nn[1]
    asset_km = val_km[1]
    count_nn = val_nn[2]
    count_km = val_nn[2]

    # portfolio sizes
    N_true = len(val_true[0])
    # retrieve fair premiums
    premiums_true = np.zeros((N_true,))
    for i in range(N_true):
        # contracts: 0: current age (!), 1: sum_ins, 2: duration, 3: elapsed duration, 4: interest rate
        premiums_true[i] = get_termlife_premium(age_init = mp_true[i,0]-mp_true[i,3], Sum_ins = mp_true[i,1], 
                                                duration = mp_true[i,2].astype('int'),  interest = mp_true[i,4], A= A, B=B, c=c)

    shock_base = target_investment_return(contracts=mp_true, assets=asset_true[0], premiums=premiums_true, A=A,B=B,c=c)

    # shocks on interest rate
    step = 0.01
    shock = np.arange(-0.05, 0.05+step/10, step).tolist()#+[shock_base]#np.arange(shock_base-5*step/10,shock_base+6*step/10,step).tolist()
    shock.sort()
    # Premiums NN
    if type(val_nn[0]) != type(None):
        K_nn = len(val_nn[0])
        premiums_nn = np.zeros((K_nn,))
        for i in range(K_nn):
            # note: respect integer-valued remaining duration and dictionary type of val_nn (!)
            nn_up = get_termlife_premium(age_init = mp_nn[i,0]-mp_nn[i,3], Sum_ins = mp_nn[i,1], 
                                                duration = np.ceil(mp_nn[i,2]).astype('int'),  interest = mp_nn[i,4], A= A, B=B, c=c)
            nn_low = get_termlife_premium(age_init = mp_nn[i,0]-mp_nn[i,3], Sum_ins = mp_nn[i,1], 
                                                duration = np.floor(mp_nn[i,2]).astype('int'),  interest = mp_nn[i,4], A= A, B=B, c=c)
            premiums_nn[i] = (nn_up+nn_low)/2

    # Premiums KM
    if type(val_km[0]) != type(None):
        K_km = len(val_km[0])
        if (K_km != K_nn):
            print('ValError: K_km != K_nn.')
            exit()
        premiums_km = np.zeros((K_km,))
        for i in range(K_km):
            # note: respect integer-valued remaining duration and dictionary type of val_nn (!)
            km_up = get_termlife_premium(age_init = (mp_km[i,0]-mp_km[i,3]), Sum_ins = mp_km[i,1], 
                                                duration = np.ceil(mp_km[i,2]).astype('int'),  interest = mp_km[i,4], A= A, B=B, c=c)
            km_low = get_termlife_premium(age_init = np.ceil(mp_km[i,0]-mp_km[i,3]), Sum_ins = mp_km[i,1], 
                                                duration = np.floor(mp_km[i,2]).astype('int'),  interest = mp_km[i,4], A= A, B=B, c=c)
            premiums_km[i] = (km_up+km_low)/2

    print('Overall premium income (True/ NN/ KM): ', premiums_true.sum(), '/ ', (premiums_nn*count_nn).sum(), ' / ', (premiums_km*count_km).sum())

    print('Fund volume (true/ NN/ KM): ', (asset_true[0].sum(), asset_true[1].sum()), ' / ', 
            ((asset_nn[0]*count_nn).sum(), (asset_nn[1]*count_nn).sum()), ' / ', ((asset_km[0]*count_km).sum(), (asset_km[1]*count_km).sum()))
    print('Agg. sum insured (true/ NN/ KM): ', mp_true[:,1].sum(), ' / ', (mp_nn[:,1]*count_nn).sum(), ' / ', (mp_km[:,1]*count_km).sum(), '\n')

    # true portfolio
    scr_true = [one_step_investment_projection(contracts = mp_true, assets= asset_true, premiums = premiums_true, shock_int = i,
                                        A= A, B= B, c= c) for i in shock]

    ## Neural network SCR
    if type(val_nn[0]) != type(None):
        scr_nn = [one_step_investment_projection(contracts = mp_nn, assets= asset_nn, premiums = premiums_nn, counts= count_nn,
                        shock_int = i,  A= A, B= B, c= c) for i in shock]
    else:
        scr_nn = [None]*len(shock)

    ## KMeans SCR
    if type(val_nn[0]) != type(None):
        scr_km = [one_step_investment_projection(contracts = mp_km, assets= asset_km, premiums = premiums_km, counts= count_km,
                        shock_int = i,  A= A, B= B, c= c) for i in shock]
    else:
        scr_km = [None]*len(shock)

    df = pd.DataFrame(data = [scr_true, scr_nn, scr_km], columns = shock, index = [r'$P$', r'$P^{(NN)}$', r'$P^{(KM)}$'] )
    print(df)
    return df


################################################################
##############   LEGACY CODE ###################################
################################################################



# def one_step_scr_projection_old(contracts, assets, counts = None, premiums = None, shock_int = 0, shock_mort = 0, 
#                             A= 0.00022, B=2.7*10**(-6), c=1.124, bool_separate = True):

#     '''
#     Come single period projection of assets and liabilites, including shock szenarios, for SCR purpose.

#     Inputs:
#     -------
#         contracts:  contracts data (term life only), i.e. age, sum_ins, duration, elapsed duration and interest rate
#         assets:     tuple: policy value per contract at times 0 and 1
#         shock_int:  additive interest rate shock; shock on assets
#         shock_mort: multiplicative shock on mortality curve; shock on liabilites
#         bool_separate: show investment and risk surplus separately

#     Outputs:
#     --------
#         tuple of 1-period-projected assets and liabilities
#     '''

#     # retrieve fair premiums, based on original assumptions at underwriting of contracts (if not provided as input yet)
#     if type(premiums) == type(None):
#         premiums = np.zeros((len(contracts),))
#         for i in range(len(contracts)):
#             # contracts: 0: current age (!), 1: sum_ins, 2: duration, 3: elapsed duration, 4: interest rate
#             premiums[i] = get_termlife_premium(age_init = contracts[i,0]-contracts[i,3], Sum_ins = contracts[i,1], duration = contracts[i,2].astype('int'),  interest = contracts[i,4], A= A, B=B, c=c)

#     # key quantities
#     interest_raw = contracts[:,4]
#     interest_shocked = interest_raw + shock_int
#     time = 1*((contracts[:,2]-contracts[:,3])>= 1) + (contracts[:,2]-contracts[:,3])*((contracts[:,2]-contracts[:,3])<1)
#     prob_death_raw = (1-np.exp(-A*time-B/np.log(c)*c**(contracts[:,0])*(c**time-1))) # 1-year death prob for current age
    
#     #print(time)
#     prob_death_shocked = prob_death_raw+shock_mort

#     # compute assets to cover losses (only affected by interest shock, not mortality)
#     inv_surplus = (assets[0]+premiums)*(interest_shocked-interest_raw)
#     # compute expected death-claims
#     risk_surplus = (contracts[:,1]-assets[1])*(prob_death_raw-prob_death_shocked)

#     if type(counts) != type(None):
#         inv_surplus *= counts
#         risk_surplus*= counts

#     if bool_separate:
#         return (inv_surplus.sum(),risk_surplus.sum())
#     else:
#         return inv_surplus.sum()+risk_surplus.sum()


# def SCR_interest_analysis_old(val_true, val_nn = (None, None), val_km = (None, None), A= 0.00022, B=2.7*10**(-6), c=1.124 ):

#     '''
#     Create table looking at investment surplus of true portfolio and grouping, in the light of the baseline szenario and interest rate shocks.

#     Inputs:
#     -------
#         val_true:     tuple: contract data (term life only), i.e. age, sum_ins, duration, elapsed duration and interest rate, and fund volume (tuple: time 0 and 1)
#         val_nn:       triplet: contract data of nn grouping, fund volume (tuple: time 0 and 1), member_counts
#         val_km:       triplet: contract data of km grouping, fund volume (tuple: time 0 and 1), member_counts 
#         *args         mortality parameters A, B, c
#     Outputs:
#     --------
#     '''

#     # unzip
#     mp_true = val_true[0]
#     mp_nn = val_nn[0]
#     mp_km = val_km[0]
#     asset_true = val_true[1]
#     asset_nn = val_nn[1]
#     asset_km = val_km[1]
#     count_nn = val_nn[2]
#     count_km = val_nn[2]


#     # shocks on interest rate
#     shock = np.arange(-0.05, 0.055, 0.005)

#     # portfolio sizes
#     N_true = len(val_true[0])
#     # retrieve fair premiums
#     premiums_true = np.zeros((N_true,))
#     for i in range(N_true):
#         # contracts: 0: current age (!), 1: sum_ins, 2: duration, 3: elapsed duration, 4: interest rate
#         premiums_true[i] = get_termlife_premium(age_init = mp_true[i,0]-mp_true[i,3], Sum_ins = mp_true[i,1], 
#                                                 duration = mp_true[i,2].astype('int'),  interest = mp_true[i,4], A= A, B=B, c=c)

#     # Premiums NN
#     if type(val_nn[0]) != type(None):
#         K_nn = len(val_nn[0])
#         premiums_nn = np.zeros((K_nn,))
#         for i in range(K_nn):
#             # note: respect integer-valued remaining duration and dictionary type of val_nn (!)
#             nn_up = get_termlife_premium(age_init = mp_nn[i,0]-mp_nn[i,3], Sum_ins = mp_nn[i,1], 
#                                                 duration = np.ceil(mp_nn[i,2]).astype('int'),  interest = mp_nn[i,4], A= A, B=B, c=c)
#             nn_low = get_termlife_premium(age_init = mp_nn[i,0]-mp_nn[i,3], Sum_ins = mp_nn[i,1], 
#                                                 duration = np.floor(mp_nn[i,2]).astype('int'),  interest = mp_nn[i,4], A= A, B=B, c=c)
#             premiums_nn[i] = (nn_up+nn_low)/2

#     # Premiums KM
#     if type(val_km[0]) != type(None):
#         K_km = len(val_km[0])
#         if (K_km != K_nn):
#             print('ValError: K_km != K_nn.')
#             exit()
#         premiums_km = np.zeros((K_km,))
#         for i in range(K_km):
#             # note: respect integer-valued remaining duration and dictionary type of val_nn (!)
#             km_up = get_termlife_premium(age_init = (mp_km[i,0]-mp_km[i,3]), Sum_ins = mp_km[i,1], 
#                                                 duration = np.ceil(mp_km[i,2]).astype('int'),  interest = mp_km[i,4], A= A, B=B, c=c)
#             km_low = get_termlife_premium(age_init = np.ceil(mp_km[i,0]-mp_km[i,3]), Sum_ins = mp_km[i,1], 
#                                                 duration = np.floor(mp_km[i,2]).astype('int'),  interest = mp_km[i,4], A= A, B=B, c=c)
#             premiums_km[i] = (km_up+km_low)/2

#     #print('Premium amount (NN/ KM): ')
#     #print('\t', premiums_nn)
#     #print('\t', premiums_km)

#     #print('# contracts: ', N_true)
#     #print('count_nn sum: ', count_nn.sum())
#     #print('count_km sum: ', count_km.sum())
#     print('Overall premium income (True/ NN/ KM): ', premiums_true.sum(), '/ ', (premiums_nn*count_nn).sum(), ' / ', (premiums_km*count_km).sum())

#     print('Fund volume (true/ NN/ KM): ', (asset_true[0].sum(), asset_true[1].sum()), ' / ', 
#             ((asset_nn[0]*count_nn).sum(), (asset_nn[1]*count_nn).sum()), ' / ', ((asset_km[0]*count_km).sum(), (asset_km[1]*count_km).sum()))
#     print('Agg. sum insured (true/ NN/ KM): ', mp_true[:,1].sum(), ' / ', (mp_nn[:,1]*count_nn).sum(), ' / ', (mp_km[:,1]*count_km).sum(), '\n')

#     # true portfolio
#     scr_true = [one_step_scr_projection_old(contracts = mp_true, assets= asset_true, premiums = premiums_true, shock_int = i, shock_mort = 0, bool_separate=False, 
#                                         A= A, B= B, c= c) for i in shock]

#     ## Neural network SCR
#     if type(val_nn[0]) != type(None):
#         scr_nn = [one_step_scr_projection_old(contracts = mp_nn, assets= asset_nn, premiums = premiums_nn, counts= count_nn, bool_separate=False,
#                         shock_int = i, shock_mort = 0, A= A, B= B, c= c) for i in shock]
#     else:
#         scr_nn = [None]*len(shock)

#     ## KMeans SCR
#     if type(val_nn[0]) != type(None):
#         scr_km = [one_step_scr_projection_old(contracts = mp_km, assets= asset_km, premiums = premiums_km, counts= count_km, bool_separate=False,
#                         shock_int = i, shock_mort = 0, A= A, B= B, c= c) for i in shock]
#     else:
#         scr_km = [None]*len(shock)

#     df = pd.DataFrame(data = [scr_true, scr_nn, scr_km], columns = shock, index = [r'$P$', r'$P^{(NN)}$', r'$P^{(KM)}$'] )
#     print(df)
#     return df


# def SCR_analysis_old(val_true, val_nn = (None, None), val_km = (None, None), A= 0.00022, B=2.7*10**(-6), c=1.124 ):

#     '''
#     Create table comparing SCR of true portfolio and grouping, in the light of the baseline szenario and mortality and interest rate shocks.

#     Inputs:
#     -------
#         val_true:     tuple: contract data (term life only), i.e. age, sum_ins, duration, elapsed duration and interest rate, and fund volume
#         val_nn:       triplet: contract data of nn grouping, fund volume, member_counts
#         val_km:       triplet: contract data of km grouping, fund volume, member_counts 
#         *args         mortality parameters A, B, c
#     Outputs:
#     --------
#     '''

#     # shocks on (interest rate/ mortality)
#     base = (0,0)
#     shock1 = (-0.01,0)
#     shock2 = (0,-0.01)
#     shock3 = (-0.006, -0.006)

#     # portfolio sizes
#     N_true = len(val_true[0])
#     # retrieve fair premiums
#     premiums_true = np.zeros((N_true,))
#     for i in range(N_true):
#         # contracts: 0: current age (!), 1: sum_ins, 2: duration, 3: elapsed duration, 4: interest rate
#         premiums_true[i] = get_termlife_premium(age_init = val_true[0][i][0]-val_true[0][i][3], Sum_ins = val_true[0][i,1], 
#                                                 duration = val_true[0][i,2].astype('int'),  interest = val_true[0][i,4], A= A, B=B, c=c)

#     if type(val_nn[0]) != type(None):
#         K_nn = len(val_nn[0])
#         premiums_nn = np.zeros((K_nn,))
#         for i in range(K_nn):
#             # note: respect integer-valued remaining duration and dictionary type of val_nn (!)
#             #nn_up = get_termlife_premium(age_init = np.ceil(val_nn[0][i,0]-val_nn[0][i,3]), Sum_ins = val_nn[0][i,1], 
#             #                                    duration = val_nn[0][i,2].astype('int'),  interest = val_nn[0][i,4], A= A, B=B, c=c)
#             nn_up = get_termlife_premium(age_init = val_nn[0][i,0]-val_nn[0][i,3], Sum_ins = val_nn[0][i,1], 
#                                                 duration = np.ceil(val_nn[0][i,2]).astype('int'),  interest = val_nn[0][i,4], A= A, B=B, c=c)
#             #             
#             #nn_low = get_termlife_premium(age_init = np.floor(val_nn[0][i,0]-val_nn[0][i,3]), Sum_ins = val_nn[0][i,1], 
#             #                                    duration = val_nn[0][i,2].astype('int'),  interest = val_nn[0][i,4], A= A, B=B, c=c)
#             nn_low = get_termlife_premium(age_init = val_nn[0][i,0]-val_nn[0][i,3], Sum_ins = val_nn[0][i,1], 
#                                                 duration = np.floor(val_nn[0][i,2]).astype('int'),  interest = val_nn[0][i,4], A= A, B=B, c=c)
#             premiums_nn[i] = (nn_up+nn_low)/2

#     if type(val_km[0]) != type(None):
#         K_km = len(val_km[0])

#         if (K_km != K_nn):
#             print('ValError: K_km != K_nn.')
#             exit()
#         premiums_km = np.zeros((K_km,))
#         for i in range(K_km):
#             # note: respect integer-valued remaining duration and dictionary type of val_nn (!)
#             #km_up = get_termlife_premium(age_init = np.ceil(val_km[0][i,0]-val_km[0][i,3]), Sum_ins = val_km[0][i,1], 
#             #                                    duration = val_km[0][i,2].astype('int'),  interest = val_km[0][i,4], A= A, B=B, c=c)
#             km_up = get_termlife_premium(age_init = (val_km[0][i,0]-val_km[0][i,3]), Sum_ins = val_km[0][i,1], 
#                                                 duration = np.ceil(val_km[0][i,2]).astype('int'),  interest = val_km[0][i,4], A= A, B=B, c=c)
#             #km_low = get_termlife_premium(age_init = np.floor(val_km[0][i,0]-val_nn[0][i,3]), Sum_ins = val_km[0][i,1], 
#             #                                    duration = val_km[0][i,2].astype('int'),  interest = val_km[0][i,4], A= A, B=B, c=c)
#             km_low = get_termlife_premium(age_init = np.ceil(val_km[0][i,0]-val_km[0][i,3]), Sum_ins = val_km[0][i,1], 
#                                                 duration = np.floor(val_km[0][i,2]).astype('int'),  interest = val_km[0][i,4], A= A, B=B, c=c)
#             premiums_km[i] = (km_up+km_low)/2

#     print('Premium amount (NN/ KM): ')
#     print('\t', premiums_nn)
#     print('\t', premiums_km)
#     print('Overall premium income (True/ NN/ KM): ', premiums_true.sum(), '/ ', (premiums_nn*val_nn[2]).sum(), ' / ', (premiums_km*val_km[2]).sum())

#     print('Fund volume (true/ NN/ KM): ', val_true[1].sum(), ' / ', (val_nn[1]*val_nn[2]).sum(), ' / ', (val_km[1]*val_km[2]).sum())
#     print('Agg. sum insured (true/ NN/ KM): ', val_true[0][:,1].sum(), ' / ', (val_nn[0][:,1]*val_nn[2]).sum(), ' / ', (val_km[0][:,1]*val_km[2]).sum(), '\n')

#     # baseline
#     scr_true_base = one_step_scr_projection_old(contracts = val_true[0], assets= val_true[1], premiums = premiums_true, shock_int = base[0], shock_mort = base[1], 
#                                         A= A, B= B, c= c).sum() 
#     # interest rate shock
#     scr_true_int = one_step_scr_projection_old(contracts = val_true[0], assets= val_true[1], premiums = premiums_true, shock_int = shock1[0], shock_mort = shock1[1], 
#                                         A= A, B= B, c= c).sum()
#     # mortality shock
#     scr_true_mort = one_step_scr_projection_old(contracts = val_true[0], assets= val_true[1], premiums = premiums_true, shock_int = shock2[0], shock_mort = shock2[1],
#                                         A= A, B= B, c= c).sum()
#     # combined shock
#     scr_true_comb = one_step_scr_projection_old(contracts = val_true[0], assets= val_true[1], premiums = premiums_true, shock_int = shock3[0], shock_mort = shock3[1],
#                                         A= A, B= B, c= c).sum()

#     ## Neural network SCR
#     if type(val_nn[0]) != type(None):
#         # baseline
#         scr_nn_base = (one_step_scr_projection_old(contracts = val_nn[0], assets= val_nn[1], premiums = premiums_nn, shock_int = base[0], shock_mort = base[1],
#                                         A= A, B= B, c= c)*val_nn[2]).sum()
#         # interest rate shock
#         scr_nn_int = (one_step_scr_projection_old(contracts = val_nn[0], assets= val_nn[1], premiums = premiums_nn, shock_int = shock1[0], shock_mort = shock1[1], 
#                                         A= A, B= B, c= c)*val_nn[2]).sum()
#         # mortality shock
#         scr_nn_mort = (one_step_scr_projection_old(contracts = val_nn[0], assets= val_nn[1], premiums = premiums_nn, shock_int = shock2[0], shock_mort = shock2[1],
#                                         A= A, B= B, c= c)*val_nn[2]).sum()
#         # interest rate shock
#         scr_nn_comb = (one_step_scr_projection_old(contracts = val_nn[0], assets= val_nn[1], premiums = premiums_nn, shock_int = shock3[0], shock_mort = shock3[1],
#                                         A= A, B= B, c= c)*val_nn[2]).sum()
#     else:
#         scr_nn_base, scr_nn_int, scr_nn_mort, scr_nn_comb = None, None, None, None


#     ## KMeans SCR
#     if type(val_nn[0]) != type(None):
#         # baseline
#         scr_km_base = (one_step_scr_projection_old(contracts = val_km[0], assets= val_km[1], premiums = premiums_km, shock_int = base[0], shock_mort = base[1],
#                                         A= A, B= B, c= c)*val_km[2]).sum()
#         # interest rate shock
#         scr_km_int = (one_step_scr_projection_old(contracts = val_km[0], assets= val_km[1], premiums = premiums_km, shock_int = shock1[0], shock_mort = shock1[1], 
#                                         A= A, B= B, c= c)*val_km[2]).sum()
#         # mortality shock
#         scr_km_mort = (one_step_scr_projection_old(contracts = val_km[0], assets= val_km[1], premiums = premiums_km, shock_int = shock2[0], shock_mort = shock2[1],
#                                         A= A, B= B, c= c)*val_km[2]).sum()
#         # interest rate shock
#         scr_km_comb = (one_step_scr_projection_old(contracts = val_km[0], assets= val_km[1], premiums = premiums_km, shock_int = shock3[0], shock_mort = shock3[1],
#                                         A= A, B= B, c= c)*val_km[2]).sum()
#     else:
#         scr_km_base, scr_km_int, scr_km_mort, scr_km_comb = None, None, None, None


#     df = pd.DataFrame(data = None, index = ['P', 'P^(NN)', 'P^(KM)'] )
#     df['base'] = [scr_true_base, scr_nn_base, scr_km_base]
#     df['shock 1'] = [scr_true_int, scr_nn_int, scr_km_int]
#     df['shock 2'] = [scr_true_mort, scr_nn_mort, scr_km_mort]
#     df['shock 3'] = [scr_true_comb, scr_nn_comb, scr_km_comb]

#     print(df)
#     return df
