import pandas as pd
import numpy as np
import copy
import warnings


def get_sticks_from_roots(roots, eshift, erange=(2400, 2500)):
    x = []
    y = []
    for num, root in roots.items():
        if root['eV'] > 0:
            continue
        energy = abs(root['eV']) + eshift
        if (energy > erange[0]) & (energy < erange[1]):
            x.append(energy)
            y.append(abs(root['Total Oscillator Strength']))
    return np.array([x, y])


def get_sticks(parsed, energyshift, erange=(2440, 2475), oscillatorthresh=1e-10):
    '''Deprecated'''
    warnings.warn('Kept for backwards compatibility, use get_sticks_from_roots instead', DeprecationWarning)
    x = []
    y = []
    for r, v in parsed.items():
        energy = abs(float(v['eV'])) + energyshift
        if energy > erange[0] and energy < erange[1]:
            if abs(v['Total Oscillator Strength']) > oscillatorthresh:
                x.append(energy)
                y.append(abs(v['Total Oscillator Strength']))
    return np.array([x, y])

def get_transitions_erange_threshold(parsed, energyshift, erange=(2440, 2475), oscillatorthresh=1e-10):
    trans = []
    for r, v in parsed.items():
        energy = abs(float(v['eV'])) + energyshift
        if energy > erange[0] and energy < erange[1]:
            if abs(v['Total Oscillator Strength']) > oscillatorthresh:
                trans.append(r)
    new = {}
    for t in trans:
        new[t] = parsed[t]
    return new


def get_transitions_orbitals_beta_only(roots, movecs, energyshift, rootthresh=0.25, bfnthresh=0.1):
    '''Deprecated'''
    warnings.warn('Deprecated in favor of get_transitions_orbitals_beta_only_normed', DeprecationWarning)
    transorbs = []
    for rnum, root in roots.items():
        for transition in root['transitions']:
            if abs(transition['coeff']) > rootthresh:
                try:
                    occm = int(transition['occ (beta)'])
                except KeyError:
                    continue
                m = movecs[occm]
                for bfn in m['Bfns']:
                    if abs(float(bfn['Coefficient'])) > bfnthresh:
                        bfn = bfn.copy()
                        bfn['osc str'] = root['Total Oscillator Strength']
                        bfn['MOnum'] = occm
                        bfn['root'] = rnum
                        bfn['root energy'] = abs(float(root['eV'])) + energyshift
                        bfn['root coeff abs.'] = abs(transition['coeff'])
                        transorbs.append(bfn)
    return pd.DataFrame(transorbs)


def get_transitions_orbitals_beta_only_normed(roots, movecs, energyshift):
    normedroots = copy.deepcopy(roots)
    for rnum, root in normedroots.items():
        transtotal = 0
        for transition in root['transitions']:
            transition['coeff^2'] = transition['coeff'] ** 2
            transtotal += transition['coeff^2']
        for transition in root['transitions']:
            transition['normedcoeff^2'] = transition['coeff^2'] / transtotal
    roots = normedroots
    
    normedmovecs = copy.deepcopy(movecs)
    new = {}
    # crop to only occupied movecs to reduce computation time
    for monum, mv in normedmovecs.items():
        if mv['Occupation'] > 0:
            new[monum] = mv
    normedmovecs = new
    # normalize
    for monum, mv in normedmovecs.items():
        mototal = 0
        for bfn in mv['Bfns']:
            mototal += abs(bfn['Coefficient'])
        for bfn in mv['Bfns']:
            bfn['NormedAbsCoefficient'] = abs(bfn['Coefficient']) / mototal
    movecs = normedmovecs
    
    transorbs = []
    for rnum, root in roots.items():
        for transition in root['transitions']:
            try:
                occm = int(transition['occ (beta)'])
            except KeyError:
                continue
            m = movecs[occm]
            for bfn in m['Bfns']:
                bfn = bfn.copy()
                bfn['OscStr'] = root['Total Oscillator Strength']
                bfn['MOnum'] = occm
                bfn['root'] = rnum
                bfn['root energy'] = abs(root['eV']) + energyshift
                bfn['root coeff^2 normed'] = transition['normedcoeff^2']
                transorbs.append(bfn)
    
    transorbs = pd.DataFrame(transorbs)
    
    transorbs['weightedcontrib'] = transorbs['NormedAbsCoefficient'] * transorbs['root coeff^2 normed'] * transorbs['OscStr'].abs()
                
    return transorbs


def calc_orbitals_fractions(orbs, erange, normalizerootcoeffs=True, normalizebfncoefficients=True):
    '''Calculate fractions of atomic orbitals contributing to transitions in a specified energy range.
    
    Normalize keywords do simple normalizations of the absolute values of the coefficients.
    '''
    orbs = orbs.copy()
    orbs = orbs[(orbs['root energy'] > erange[0]) & (orbs['root energy'] < erange[1])]
    
    # normalize root coefficients for each root
    if normalizerootcoeffs:
        for r in orbs['root'].unique():
            sumrootcoeff = 0
            for mo in orbs[orbs['root']==r]['MOnum'].unique():
                sumrootcoeff += orbs[(orbs['root']==r) & (orbs['MOnum']==mo)]['root coeff abs.'].iloc[0]

            orbs.loc[orbs['root']==r, 'root coeff abs.'] /= sumrootcoeff

    # normalize coefficient of bfn's for each MO
    if normalizebfncoefficients:
        for r in orbs['root'].unique():
            for mo in orbs[orbs['root']==r]['MOnum'].unique():
                subset = (orbs['root']==r) & (orbs['MOnum']==mo)
                orbs.loc[subset, 'Coefficient'] /= orbs.loc[subset, 'Coefficient'].abs().sum()
    
    fractions = {}
    for i, row in orbs.iterrows():
        try:
            fractions['{} {}'.format(row['Atom'], row['Atom Fn.'][0])] += abs(row['Coefficient']) * abs(row['root coeff abs.']) * abs(row['osc str'])
        except KeyError:
            fractions['{} {}'.format(row['Atom'], row['Atom Fn.'][0])] = 0.
            fractions['{} {}'.format(row['Atom'], row['Atom Fn.'][0])] += abs(row['Coefficient']) * abs(row['root coeff abs.']) * abs(row['osc str'])
            
    normedfracs = {}
    for k, f in fractions.items():
        normedfracs[k] = f / np.sum(list(fractions.values()))
    
    return normedfracs

def check_vector_direction(vec, direction, threshold):
    return abs(np.dot(vec, direction)) / (abs(np.linalg.norm(vec)) * abs(np.linalg.norm(direction))) >= threshold

def filter_roots_by_tm_direction(roots, direction, threshold):
    filtered = {}
    for r, v in roots.items():
        if check_vector_direction(v['Transition Moments (XYZ)'], direction, threshold):
            filtered[r] = v
    return filtered


def calc_normalized_root_contributions(orbs, erange, normalizerootcoeffs=True, normalizebfncoefficients=True):
    '''Calculate fractions of atomic orbitals contributing to transitions in a specified energy range.
    
    Normalize keywords do simple normalizations of the absolute values of the coefficients.
    '''
    orbs = orbs.copy()
    orbs = orbs[(orbs['root energy'] > erange[0]) & (orbs['root energy'] < erange[1])]
    
    # normalize root coefficients for each root
    if normalizerootcoeffs:
        for r in orbs['root'].unique():
            sumrootcoeff = 0
            for mo in orbs[orbs['root']==r]['MOnum'].unique():
                sumrootcoeff += orbs[(orbs['root']==r) & (orbs['MOnum']==mo)]['root coeff abs.'].iloc[0]

            orbs.loc[orbs['root']==r, 'root coeff abs.'] /= sumrootcoeff

    # normalize coefficient of bfn's for each MO
    if normalizebfncoefficients:
        for r in orbs['root'].unique():
            for mo in orbs[orbs['root']==r]['MOnum'].unique():
                subset = (orbs['root']==r) & (orbs['MOnum']==mo)
                orbs.loc[subset, 'Coefficient'] /= orbs.loc[subset, 'Coefficient'].abs().sum()
    
    new = orbs.copy()
    for i, row in orbs.iterrows():
        new.loc[i, 'weightedcontrib'] = abs(row['Coefficient']) * abs(row['root coeff abs.']) * abs(row['osc str'])

    return new


def get_transition_moment_projected_strength(roots, direction):
    projected = {}
    for r, v in roots.items():
        assert abs(v['Electric Quadrupole']) < 1e-4, 'Large quadrupole found, but projection based on dipole-behavior.'
        
        
        tmvec = v['Transition Moments (XYZ)']
        tmvec = tmvec.copy()
        norm = np.linalg.norm(tmvec)
        if norm > 0:
            tmvec /= np.linalg.norm(tmvec)
        direction /= np.linalg.norm(direction)
        proj = v['Total Oscillator Strength'] * np.dot(tmvec, direction) ** 2
        if not np.isnan(proj):
            projected[r] = {}
            projected[r]['ProjOscStr'] = proj
            projected[r]['eV'] = v['eV']
            projected[r]['ProjDir'] = direction
    return projected


def calc_orbitals_fractions_from_transitions_orbitals(transorbs, erange, fracthresh=0.05):
    transorbs = copy.deepcopy(transorbs[(transorbs['root energy'] >= erange[0]) & (transorbs['root energy'] < erange[1])])
    
    fracs = {}
    total = 0
    for i, row in transorbs.iterrows():
        if row['Atom'] not in fracs:
            fracs[row['Atom']] = {}
        if row['Atom Fn.'] not in fracs[row['Atom']]:
            fracs[row['Atom']][row['Atom Fn.']] = 0
        fracs[row['Atom']][row['Atom Fn.']] += row['weightedcontrib']
        total += row['weightedcontrib']

    # normalize
    new = copy.deepcopy(fracs)
    for atom, d in fracs.items():
        for fn, v in d.items():
            new[atom][fn] = v / total
    fracs = new

    # threshold
    new = {}
    for atom, d in fracs.items():
        new[atom] = {}
        for fn, v in d.items():
            if v > fracthresh:
                new[atom][fn] = v
    
    return new

def rotation_matrix_to_align_vectors(fromvec, tovec):
    '''Based on:
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    '''
    fromvec = fromvec / np.linalg.norm(fromvec)
    tovec = tovec / np.linalg.norm(tovec)
    vcross = np.cross(fromvec, tovec)
    cos = np.dot(fromvec, tovec)
    skewm = np.array([[0, -vcross[2], vcross[1]], 
                      [vcross[2], 0, -vcross[0]], 
                      [-vcross[1], vcross[0], 0]])
    rotM = np.identity(3) + skewm + np.matmul(skewm, skewm) * (1 / (1 + cos))
    return rotM
