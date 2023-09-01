import re
import os
import time
from subprocess import call
import numpy as np
#import matplotlib.pyplot as plt


def replace_text_in_file(infile, oldstr, newstr):
    # Read in the file
    with open(infile, 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace(oldstr, newstr)

    # Write the file out again
    with open(infile, 'w') as file:
        file.write(filedata)


def find_highest_number_xyz_file(directory):
    numbers = []
    prefixes = []
    for filename in os.listdir(directory):
        match = re.match(r'(.*?)(\d*)(.xyz)', filename)
        prefixes.append(match.group(1))
        numbers.append(match.group(2))
    assert all([p==prefixes[0] for p in prefixes]), 'Some filename prefixes do not match'
    numbers = [int(n) for n in numbers]
    return '{}{:03d}{}'.format(match.group(1), max(numbers), match.group(3))


def read_last_line(file):
#https://stackoverflow.com/questions/3346430/what-is-the-most-efficient-way-to-get-first-and-last-line-of-a-text-file
    with open(file, "rb") as f:
        f.readline()         # Read the first line.
        f.seek(-2, os.SEEK_END)      # Jump to the second last byte.
        while f.read(1) != b"\n":    # Until EOL is found...
            f.seek(-2, os.SEEK_CUR)  # ...jump back the read byte plus one more.
        last = f.readline()          # Read last line.
    return last


def check_calculation_successful(outfile):
    try:
        res = read_last_line(outfile)[:12] == b' Total times'
        return res
    except IOError:
        return False


def get_highest_occupied_beta_movec(infile): #Deprecated, kept for backwards compatibility
    with open(infile, 'r') as f:
        content = f.read()
        betaorbitalsindex = content.index('DFT Final Beta Molecular Orbital Analysis')
        betaorbitals = content[betaorbitalsindex:]
        occ0index = betaorbitals.index('Occ=0')
        f.seek(betaorbitalsindex + occ0index)
        vectorindex = betaorbitals.index('Vector', occ0index - 14, occ0index)
        f.seek(betaorbitalsindex + vectorindex)
        r = f.readline()
    return int(r.split()[1]) - 1


def get_highest_occupied_movec(infile, channel='beta'):
    if channel == 'beta':
        channel = 'Beta'
    elif channel == 'alpha':
        channel = 'Alpha'
    else:
        raise RuntimeError('Channel must be \'alpha\' or \'beta\'')
    with open(infile, 'r') as f:
        content = f.read()
        orbitalsindex = content.index('DFT Final {} Molecular Orbital Analysis'.format(channel))
        orbitals = content[orbitalsindex:]
        occ0index = orbitals.index('Occ=0')
        f.seek(orbitalsindex + occ0index)
        vectorindex = orbitals.index('Vector', occ0index - 14, occ0index)
        f.seek(orbitalsindex + vectorindex)
        r = f.readline()
    return int(r.split()[1]) - 1


def get_number_alphas_betas(infile):
    res = {}
    with open(infile, 'r') as f:
        content = f.read()
        generalinfo = content[content.index('General Information') : content.index('XC Information')]
        res['alphas'] = int([s for s in generalinfo.splitlines() if 'Alpha' in s][0].split(':')[1])
        res['betas'] = int([s for s in generalinfo.splitlines() if 'Beta' in s][0].split(':')[1])
    return res


def start_job():
    return call(['msub', 'job.run'])


def wait_for_calculation_completion(outfilename, maxwait=3600*24):
    w = 0
    while not os.path.isfile(outfilename):
        print('waiting for job to start: {}s'.format(w), end='\r')
        w = w + 1
        time.sleep(1)

    print('output file created. blocking for maxwait {}s'.format(maxwait))
    i = 0
    while not check_calculation_successful(outfilename):
        if i < maxwait:
            time.sleep(1)
            i = i + 1
        else:
            raise RuntimeError('Output file not created within {}s timeout period.'.format(maxwait))


def convert_mol_to_xyz(infile):
    name = infile.split('.mol')[0]
    with open(infile, 'r') as file:
        lines = file.readlines()

    istart = 0
    for l in lines:
        if 'V2000' in l.split() or 'V3000' in l.split():
            break
        else:
            istart = istart + 1

    atomre = re.compile(r'\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\w+)\s+')
    atoms = []
    for l in lines[istart + 1:]:
        m = atomre.match(l)
        if m is not None:
            atoms.append(m.groups())
        else:
            break

    with open(name + '.xyz', 'w') as outfile:
        outfile.write(str(len(atoms)) + '\n')
        outfile.write('File made automatically with Python script using input mol file. (v1)\n')
        for a in atoms:
            outfile.write('{}{:>15}{:>15}{:>15}\n'.format(a[-1], a[0], a[1], a[2]))


def center_xyz(infile, targetline):
    with open(infile, 'r') as file:
        with open(infile.split('.xyz')[0] + '_centered.xyz', 'w') as newfile:
            for i in range(targetline - 1):
                file.readline()
            target = file.readline().split()
            for i in range(1, len(target)):
                target[i] = float(target[i])

            file.seek(0)

            newfile.write(file.readline())
            newfile.write(file.readline())
            while True:
                r = file.readline().split()
                if len(r) != 4:
                    break
                for i in range(1, len(r)):
                    r[i] = float(r[i]) - target[i]
                newfile.write('{}  {:>15.5f}{:>15.5f}{:>15.5f}\n'.format(*r))
            newfile.write('\n')
    return infile.split('.xyz')[0] + '_centered.xyz'


def read_xyz(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        atoms = []
        coords = []
        for l in lines[2:]: #XYZ files must have two header rows
            split = l.split()
            atoms.append(split[0])
            coords.append([float(x) for x in split[1:]])
        assert len(atoms) == len(coords), 'Something went wrong, len of atoms doesnt equal length of coords'
    return atoms, coords


def basic_multiplicity_from_atoms(atoms):
    import periodictable
    electrons = 0
    for a in atoms:
        electrons += periodictable.__getattribute__(a).number
    print('{} electrons, which means basic multiplicity {}'.format(electrons, electrons % 2 + 1))
    return electrons % 2 + 1


def make_xyz_animation(basename, directory=None):
    if directory is None:
        directory = os.getcwd() + '/'
    
    filere = re.compile(r'{}\d\d\d.xyz'.format(basename))
    
    xyzfiles = []
    for f in os.listdir(directory):
        if filere.match(f) is not None:
            xyzfiles.append(f)
    
    if not xyzfiles:
        raise FileNotFoundError('No files matching {}###.xyz were found.'.format(basename))
        
    assert not os.path.exists(directory+'{}animation.xyz'.format(basename)), 'File already exists'
        
    with open(directory+'{}animation.xyz'.format(basename), 'w') as outfile:
        for xyz in sorted(xyzfiles):
            with open(directory+xyz, 'r') as infile:
                outfile.write(infile.read())



def read_dft_transitions_file(path):
    return np.loadtxt(path).T


#https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
def Lorentzian(x, xc, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / ((x-xc)**2 + gamma**2)


def spectrum_from_transitions(transitions, lorentz_ev=1, erange=None, numpoints=1000, peaknorm=True):
    x, y = transitions
    if erange is not None:
        good = np.logical_and(x >= erange[0], x <= erange[1])
        x, y = x[good], y[good]
        x_eval = np.linspace(erange[0], erange[1], numpoints)
    else:
        xmin = np.min(x)
        xmax = np.max(x)
        padding = (xmax - xmin) / 2
        x_eval = np.linspace(xmin - padding, xmax + padding, numpoints)
    
    spectrum = np.zeros_like(x_eval)
    for e, a in zip(x, y):
        spectrum += a * Lorentzian(x_eval, e, lorentz_ev/2)
    
    if peaknorm:
        spectrum = spectrum / np.max(spectrum)

    return np.array([x_eval, spectrum])


#Thank you stackoverflow
#https://stackoverflow.com/questions/24143320/gaussian-sum-filter-for-irregular-spaced-points
def gaussian_broaden(spectrum, width_ev=2, numpoints=1000, xmin=None, xmax=None):
    x, y = spectrum
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    x_eval = np.linspace(xmin, xmax, numpoints)
    sigma = width_ev/(2*np.sqrt(2*np.log(2)))

    delta_x = x_eval[:, None] - x
    weights = np.exp(-delta_x*delta_x / (2*sigma*sigma)) / (np.sqrt(2*np.pi) * sigma)
    weights /= np.sum(weights, axis=1, keepdims=True)
    y_eval = np.dot(weights, y)

    return np.array([x_eval, y_eval])


def plot_spectrum_and_transitions(transitions, lorentz_ev=1, erange=None, 
                                numpoints=1000, gaussian_ev=None, show=True):
    import matplotlib.pyplot as plt
    spectrum = spectrum_from_transitions(transitions, lorentz_ev=lorentz_ev, 
                            erange=erange, numpoints=numpoints, peaknorm=False)
    x, y = spectrum
    norm = np.max(y)

    fig, ax = plt.subplots()
    ax.plot(x, y / norm)

    #rescale so that stem matches spectral height
    rescale = Lorentzian(0, 0, lorentz_ev/2)
    xs, ys = transitions
    if erange is not None:
        good = np.logical_and(xs >= erange[0], xs <= erange[1])
        xs, ys = xs[good], ys[good]
    markerline, stemlines, baseline = ax.stem(xs, ys * rescale / norm, 
                                                basefmt='k', linefmt='C0-')
    plt.setp(baseline, visible=False)
    plt.setp(stemlines, 'linewidth', 1)
    plt.setp(markerline, 'markersize', 3)
    plt.xlabel('Energy (eV)')

    if show:
        plt.show()

    return fig


# def parse_roots_from_tddft_output(file):
#     with open(file, 'r') as f:
#         lines = f.readlines()
        
#     rootre = re.compile('Root\s+\d+')
#     transitionre = re.compile(r'\s+Occ\.\s+(\d+)\s+(alpha|beta)\s+\w\s+-+\s+Virt.\s+(\d+)\s+(alpha|beta)\s+\w\s+(-?\d+\.\d+)\s+')
    
#     rootlinenums = []
#     for i, l in enumerate(lines):
#         if rootre.search(l):
#             rootlinenums.append(i)
            
#     rootdict = {}
#     for i in range(len(rootlinenums)):
#         rootdict[i + 1] = {}
        
#     for k, d in rootdict.items():
#         d['transitions'] = []
#         if k != len(rootdict):
#             lower = rootlinenums[k-1]
#             upper = rootlinenums[k]
#         else:
#             lower = rootlinenums[k-1]
#             upper = None
#         for l in lines[lower:upper]:
#             m = transitionre.match(l)
#             if m:
#                 trans = {}
#                 trans['occ ({})'.format(m.group(2))] = m.group(1)
#                 trans['virt ({})'.format(m.group(4))] = m.group(3)
#                 trans['coeff'] = float(m.group(5))
#                 d['transitions'].append(trans)

#             dm = re.match(r'\s+Dipole Oscillator Strength\s+(-?\d+\.\d+)\s+', l)
#             if dm:
#                 d['Dipole Oscillator Strength'] = float(dm.group(1))

#             qm = re.match(r'\s+Electric Quadrupole\s+(-?\d+\.\d+)\s+', l)
#             if qm:
#                 d['Electric Quadrupole'] = float(qm.group(1))

#             mm = re.match(r'\s+Magnetic Dipole\s+(-?\d+\.\d+)\s+', l)
#             if mm:
#                 d['Magnetic Dipole'] = float(mm.group(1))

#             tm = re.match(r'\s+Total Oscillator Strength\s+(-?\d+\.\d+)\s+', l)
#             if tm:
#                 d['Total Oscillator Strength'] = float(tm.group(1))

#             auev = re.match(r'\s+Root\s+\d+\s+\w\s+(-?\d+\.\d+)\s+a.u.\s+(-?\d+\.\d+)\s+eV\s+', l)
#             if auev:
#                 d['a.u.'] = auev.group(1)
#                 d['eV'] = auev.group(2)

#             s2 = re.match(r'\s+<S2>\s+=\s+(-?\d+\.\d+)\s+', l)
#             if s2:
#                 d['<S2>'] = s2.group(1)
                
#     return rootdict


def write_xyz_from_atoms_coords(filename, atoms, coords, comment=None):
    if '.xyz' not in filename:
        filename += '.xyz'
    with open(filename, 'w') as outfile:
        outfile.write(str(len(atoms)) + '\n')
        outfile.write('{}\n'.format(comment))
        for a, c in zip(atoms, coords):
            outfile.write('{}{:>15.5f}{:>15.5f}{:>15.5f}\n'.format(a, *c))


def take_erange(energies, intensities, erange):
    good = np.where(np.logical_and(energies < erange[1], energies > erange[0]))
    return np.array([energies[good], intensities[good]])


def integral_normalize(energies, intensities, erange):
    e, i = take_erange(energies, intensities, erange)
    return np.array([energies, intensities / np.sum(i)])


def integral_norm_over_erange(x, y, erange):
    good = np.where(np.logical_and(x>erange[0], x<erange[1]))
    y = y / np.abs(np.trapz(y[good], x[good]))
    return np.array([x, y])


def proper_spectrum_from_transitions(transitions, lorentz_ev=1, erange=None, numpoints=1000, peaknorm=True):
    x, y = transitions
    if erange is not None:
        good = np.logical_and(x >= erange[0], x <= erange[1])
        x, y = x[good], y[good]
        x_eval = np.linspace(erange[0], erange[1], numpoints)
    else:
        xmin = np.min(x)
        xmax = np.max(x)
        padding = (xmax - xmin) / 2
        x_eval = np.linspace(xmin - padding, xmax + padding, numpoints)
    
    spectrum = np.zeros_like(x_eval)
    
    # Charles Note: I don't know where this comes from,
    # but overall it doesn't produce a noticably different result
    # from the regualar `spectrum_from_transitions` function
    
    #do proper energy scaling
    sy = x ** 2 * y
    y = sy / sy.sum()
    
    for e, a in zip(x, y):
        spectrum += a * Lorentzian(x_eval, e, lorentz_ev/2)
    
    if peaknorm:
        spectrum = spectrum / np.max(spectrum)

    return np.array([x_eval, spectrum])
