import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d


eV = sc.electron_volt
mp = sc.proton_mass
ry = sc.value("Rydberg constant times hc in eV")


def Maxwell_velocity_dist(v, T, m):
    Tj = T * eV
    return np.sqrt(2/np.pi)*np.sqrt(m/Tj)**3*v**2*np.exp(-m*v**2/2/Tj)


def Relative_to_dirac_dist(v, T, m, v0):
    w = np.sqrt(2*T*eV/m)
    return 1/(np.sqrt(np.pi)*w*v0)*v*(np.exp(-(v-v0)**2/w**2)-np.exp(-(v+v0)**2/w**2))


def calculate_rate(v, sigma, f):
    return np.trapz(v*sigma*f, v)


class janev_1887_H_He_CX:

    def __init__(self, n, m):
        #if n < 2:
        #    return ValueError("n must be greater than 1")
        self.n = n
        self.m = m
        self.rcoeff = [0.75,0.8,0.85,0.9,0.95,0.98]
        if (m == (2*n-1)):
            self.eth = ry * (1.0/(m*m) - 1.0/((m+1.0)**2))
        else:
            self.eth = 0.0

    def cross_section(self, pe):

        #if (pe < self.eth):
        #    return 0.0

        if (self.m == (2*self.n-1)):
            if (self.n > 7):
                const = 1.0
            else:
                const = self.rcoeff[self.n-2]
        else:
            const = 1.0

        v=3.1623e-03 * np.sqrt(pe)
        nv = self.n * v

        return 4.69e-16*const * (self.n**4) / (1.0 + 0.8*(nv**0.4) + 2.6*nv ) / 10000
    

class CrossSection:

    def __init__(self, projectile, target, product, file):
        self.projectile = projectile
        self.target = target
        self.product = product
        self.raw = self.reader(file)
        self.cross_section_m2 = self.raw[:,1] / 10000       # cm2 to m2
        if self.projectile == 'electron':
            self.energy = self.raw[:,0] * 1836.152          # energy to eV/amu if electron energy is given
        else:
            self.energy = self.raw[:,0]                     # energy in file should be eV/amu otherwise

        self._cross_section_interpolator = interp1d(np.log(self.energy), np.log(self.cross_section_m2), fill_value='extrapolate')

    def cross_section(self, value):
        return np.exp(self._cross_section_interpolator(np.log(value)))
    
    def __str__(self):
        return f'{self.projectile}, {self.target} -> {self.product}'
    
    def reader(self, file):
        E_list = []
        cs_list = []
        with open(file, 'r') as f:
            for line in f.readlines():
                if line[0] == '#':
                    continue
                else:
                    E,cs = line.split()
                    E_list.append(float(E))
                    if ':' in cs:
                        cs = cs.split(':')[0]
                    cs_list.append(float(cs))
        return np.vstack((np.array(E_list),np.array(cs_list))).T
    

class CrossSectionCollection:

    def __init__(self, cross_sections):
        self.cross_sections = cross_sections

    def get_cross_section(self, projectile, target, product):
        for cs in self.cross_sections:
            if cs.projectile == projectile and cs.target == target and cs.product == product:
                return cs



class PCX_run:

    def __init__(self, n_plasma, T_plasma, n_H, T_H, E_He, n_He0, cross_sections, dt, N, m_plasma=1, m_H=1):
        
        self.n_plasma = n_plasma
        self.T_plasma = T_plasma
        self.m_plasma = m_plasma
        self.n_H = n_H
        self.T_H = T_H
        self.m_H = m_H
        self.E_He = E_He
        self.cross_sections = cross_sections
        self.dt = dt
        self.N = N
        self.time = np.arange(N) * dt

        self.v_He = np.sqrt(2*E_He*eV/mp/4)

        self.vth_plasma = np.sqrt(2*T_plasma*eV/(self.m_plasma*mp))
        self.vth_H = np.sqrt(2*T_H*eV/(self.m_H*mp))
        self.vth_electron = np.sqrt(2*T_plasma*eV/mp * 1836.152)

        self.vmin = self.v_He - 3 * self.vth_plasma
        self.vmax = self.v_He + 3 * self.vth_plasma
        self.v = np.linspace(self.vmin, self.vmax, 1000)
        self.E_v = (0.5 * mp * self.v**2)/eV

        self.vmin_H = self.v_He - 3 * self.vth_H
        self.vmax_H = self.v_He + 3 * self.vth_H
        self.v_H = np.linspace(self.vmin_H, self.vmax_H, 1000)
        self.E_v_H = (0.5 * mp * self.v_H**2)/eV

        self.vmin_e = max(0.01, self.v_He - 3 * self.vth_electron)
        self.vmax_e = self.v_He + 3 * self.vth_electron
        self.v_e = np.linspace(self.vmin_e, self.vmax_e, 10000)
        self.E_v_e = (0.5 * mp * self.v_e**2)/eV

        self.f_plasma = Relative_to_dirac_dist(self.v, self.T_plasma, self.m_plasma*mp, self.v_He)
        self.f_H = Relative_to_dirac_dist(self.v_H, self.T_H, self.m_H*mp, self.v_He)
        self.f_e = Relative_to_dirac_dist(self.v_e, self.T_plasma, mp/1836.152, self.v_He)

        #from He+ to He2+
        self.R_He1_e_ion = calculate_rate(self.v_e, self.cross_sections.get_cross_section('electron', 'He+', 'He2+').cross_section(self.E_v_e), self.f_e)
        self.R_He1_p_loss = calculate_rate(self.v, self.cross_sections.get_cross_section('H+', 'He+', 'ionHe2+').cross_section(self.E_v), self.f_plasma) + \
                                            calculate_rate(self.v, self.cross_sections.get_cross_section('H+', 'He+', 'cxHe2+').cross_section(self.E_v), self.f_plasma)
        self.R_He1_H_ion = calculate_rate(self.v_H, self.cross_sections.get_cross_section('H', 'He+', 'He2+').cross_section(self.E_v_H), self.f_H)

        #from He2+ to He+
        self.R_He2_H_cx = calculate_rate(self.v_H, self.cross_sections.get_cross_section('H', 'He2+', 'He+').cross_section(self.E_v_H), self.f_H)

        #from He+ to He
        self.R_He1_H_cx = calculate_rate(self.v_H, self.cross_sections.get_cross_section('H', 'He+', 'He').cross_section(self.E_v_H), self.f_H)

        #from He to He+
        self.R_He_e_ion = calculate_rate(self.v_e, self.cross_sections.get_cross_section('electron', 'He', 'He+').cross_section(self.E_v_e), self.f_e)
        self.R_He_p_loss = calculate_rate(self.v, self.cross_sections.get_cross_section('H+', 'He', 'ionHe+').cross_section(self.E_v), self.f_plasma) + \
                                            calculate_rate(self.v, self.cross_sections.get_cross_section('H+', 'He', 'cxHe+').cross_section(self.E_v), self.f_plasma)
        self.R_He_H_ion = calculate_rate(self.v_H, self.cross_sections.get_cross_section('H', 'He', 'He+').cross_section(self.E_v_H), self.f_H)

        self.n_He2 = np.empty(self.N, dtype=float)
        self.n_He2[0] = n_He0
        self.n_He1 = np.empty(self.N, dtype=float)
        self.n_He1[0] = 0.0
        self.n_He = np.empty(self.N, dtype=float)
        self.n_He[0] = 0.0

        self.R_He1_to_He2 = self.n_plasma*self.R_He1_e_ion + self.n_plasma*self.R_He1_p_loss + self.n_H*self.R_He1_H_ion
        self.R_He_to_He1 = self.n_plasma*self.R_He_e_ion + self.n_plasma*self.R_He_p_loss + self.n_H*self.R_He_H_ion
        self.R_He2_to_He1 = self.n_H*self.R_He2_H_cx
        self.R_He1_to_He = self.n_H*self.R_He1_H_cx

        for i in range(1, self.N):

            He1_to_He2 = self.R_He1_to_He2*self.n_He1[i-1]
            He2_to_He1 = self.R_He2_to_He1*self.n_He2[i-1]
            He1_to_He = self.R_He1_to_He*self.n_He1[i-1]
            He_to_He1 = self.R_He_to_He1*self.n_He[i-1]

            dn_He1 = He_to_He1 + He2_to_He1 - He1_to_He - He1_to_He2
            self.n_He1[i] = min(max(self.n_He1[i-1] + dn_He1*self.dt, 0), n_He0)

            dn_He2 = He1_to_He2 - He2_to_He1
            self.n_He2[i] = min(max(self.n_He2[i-1] + dn_He2*self.dt, 0), n_He0)

            dn_He = He1_to_He - He_to_He1
            self.n_He[i] = min(max(self.n_He[i-1] + dn_He*self.dt, 0), n_He0)


