import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d


eV = sc.electron_volt
mp = sc.proton_mass
Ae = 5.486e-4


def Maxwell_velocity_dist(v, T, m):
    Tj = T * eV
    return np.sqrt(2/np.pi)*np.sqrt(m/Tj)**3*v**2*np.exp(-m*v**2/2/Tj)


def Relative_to_dirac_dist(v, T, m, v0):
    w = np.sqrt(2*T*eV/m)
    return 1/(np.sqrt(np.pi)*w*v0)*v*(np.exp(-(v-v0)**2/w**2)-np.exp(-(v+v0)**2/w**2))


def calculate_rate_coeff(v, sigma, f):
    return np.trapz(v*sigma*f, v)
    

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
            
class VelocityDistribution:

    def __init__(self, T, A, v0 = 0):
        self.T = T
        self.A = A
        self.m = A*mp
        self.v0 = v0
        self.v_thermal = np.sqrt(2*self.T*eV/self.m)

        self.vmin = max(1e-6, self.v0 - 3 * self.v_thermal)
        self.vmax = self.v0 + 3 * self.v_thermal
        self.v_values = np.linspace(self.vmin, self.vmax, 1000)
        self.E_values = (0.5 * self.m * self.v_values**2)/eV

        if self.v0 == 0:
            self.f_v = Maxwell_velocity_dist(self.v_values, self.T, self.m)
        else:
            self.f_v = Relative_to_dirac_dist(self.v_values, self.T, self.m, v0)
        self.f_E = self.f_v / self.v_values / self.m * eV


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
        self.R_He1_e_ion = calculate_rate_coeff(self.v_e, self.cross_sections.get_cross_section('electron', 'He+', 'He2+').cross_section(self.E_v_e), self.f_e)
        self.R_He1_p_loss = calculate_rate_coeff(self.v, self.cross_sections.get_cross_section('H+', 'He+', 'ionHe2+').cross_section(self.E_v), self.f_plasma) + \
                                            calculate_rate_coeff(self.v, self.cross_sections.get_cross_section('H+', 'He+', 'cxHe2+').cross_section(self.E_v), self.f_plasma)
        self.R_He1_H_ion = calculate_rate_coeff(self.v_H, self.cross_sections.get_cross_section('H', 'He+', 'He2+').cross_section(self.E_v_H), self.f_H)

        #from He2+ to He+
        self.R_He2_H_cx = calculate_rate_coeff(self.v_H, self.cross_sections.get_cross_section('H', 'He2+', 'He+').cross_section(self.E_v_H), self.f_H)

        #from He+ to He
        self.R_He1_H_cx = calculate_rate_coeff(self.v_H, self.cross_sections.get_cross_section('H', 'He+', 'He').cross_section(self.E_v_H), self.f_H)

        #from He to He+
        self.R_He_e_ion = calculate_rate_coeff(self.v_e, self.cross_sections.get_cross_section('electron', 'He', 'He+').cross_section(self.E_v_e), self.f_e)
        self.R_He_p_loss = calculate_rate_coeff(self.v, self.cross_sections.get_cross_section('H+', 'He', 'ionHe+').cross_section(self.E_v), self.f_plasma) + \
                                            calculate_rate_coeff(self.v, self.cross_sections.get_cross_section('H+', 'He', 'cxHe+').cross_section(self.E_v), self.f_plasma)
        self.R_He_H_ion = calculate_rate_coeff(self.v_H, self.cross_sections.get_cross_section('H', 'He', 'He+').cross_section(self.E_v_H), self.f_H)

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


class PCX_run_dev:

    def __init__(self, n_plasma, T_plasma, n_H, T_H, E_He, n_He0, cross_sections, z=None, dt=None, N=None, m_plasma=1, m_H=1):
        
        self.n_plasma = n_plasma
        self.T_plasma = T_plasma
        self.m_plasma = m_plasma
        self.n_H = n_H
        self.T_H = T_H
        self.m_H = m_H
        self.E_He = E_He
        self.v_He = np.sqrt(2*E_He*eV/mp/4)
        self.cross_sections = cross_sections
        if dt is None and N is None and z is not None:
            self.dist = z
            self.dt = (z[1] - z[0]) / self.v_He
            self.N = len(z)
            self.time = np.arange(self.N) * self.dt
        elif z is None and dt is not None and N is not None:
            self.dt = dt
            self.N = N
            self.time = np.arange(N) * self.dt
            self.dist = self.v_He * self.time
        else:
            print('Either provide a spatial grid (z) or number of points (N) with timestep (dt).')
            return None

        
        self.v_ion = VelocityDistribution(self.T_plasma, self.m_plasma, v0 = self.v_He)
        self.v_electron = VelocityDistribution(self.T_plasma, Ae, v0 = self.v_He)
        self.v_cloud = VelocityDistribution(self.T_H, self.m_H, v0 = self.v_He)

        self.calculate_rate_coefficients()
        self.assemble_rate_coefficient_matrix()
        self.get_reaction_masks()
        

        self.n_He = np.empty((7,self.N), dtype=float)
        self.n_He[:, 0] = 0.0
        self.n_He[-1,0] = n_He0
        n_reactants = np.vstack((5*[self.n_plasma], 5*[self.n_plasma], 5*[self.n_H], 5*[self.n_H]))

        for i in range(1, self.N):

            dn_He = np.zeros(len(self.reaction_masks), dtype=float)
            for j,mask in enumerate(self.reaction_masks):
                dn_He[j] = self.n_He[:,i-1].dot((mask*self.rate_coeff_matrix).dot(n_reactants[:,i-1]))

            self.n_He[:, i] = np.clip(self.n_He[:, i-1] + dn_He*self.dt, a_min=0, a_max=n_He0)




    def calculate_rate_coefficients(self):

        #from He+ to He2+
        self.R_He1_e_ion = calculate_rate_coeff(self.v_electron.v_values, self.cross_sections.get_cross_section('electron', 'He+', 'He2+').cross_section(self.v_electron.E_values), self.v_electron.f_v)
        self.R_He1_p_loss = calculate_rate_coeff(self.v_ion.v_values, self.cross_sections.get_cross_section('H+', 'He+', 'ionHe2+').cross_section(self.v_ion.E_values), self.v_ion.f_v) + \
                                            calculate_rate_coeff(self.v_ion.v_values, self.cross_sections.get_cross_section('H+', 'He+', 'cxHe2+').cross_section(self.v_ion.E_values), self.v_ion.f_v)
        self.R_He1_H_ion = calculate_rate_coeff(self.v_cloud.v_values, self.cross_sections.get_cross_section('H', 'He+', 'He2+').cross_section(self.v_cloud.E_values), self.v_cloud.f_v)

        #from He2+ to He+
        self.R_He2_H_cx = calculate_rate_coeff(self.v_cloud.v_values, self.cross_sections.get_cross_section('H', 'He2+', 'He+').cross_section(self.v_cloud.E_values), self.v_cloud.f_v)

        #from He+ to He
        self.R_He1_H_cx = calculate_rate_coeff(self.v_cloud.v_values, self.cross_sections.get_cross_section('H', 'He+', 'He').cross_section(self.v_cloud.E_values), self.v_cloud.f_v)

        #from He to He+
        self.R_He_e_ion = calculate_rate_coeff(self.v_electron.v_values, self.cross_sections.get_cross_section('electron', 'He', 'He+').cross_section(self.v_electron.E_values), self.v_electron.f_v)
        self.R_He_p_loss = calculate_rate_coeff(self.v_ion.v_values, self.cross_sections.get_cross_section('H+', 'He', 'ionHe+').cross_section(self.v_ion.E_values), self.v_ion.f_v) + \
                                            calculate_rate_coeff(self.v_ion.v_values, self.cross_sections.get_cross_section('H+', 'He', 'cxHe+').cross_section(self.v_ion.E_values), self.v_ion.f_v)
        self.R_He_H_ion = calculate_rate_coeff(self.v_cloud.v_values, self.cross_sections.get_cross_section('H', 'He', 'He+').cross_section(self.v_cloud.E_values), self.v_cloud.f_v)

        self.R_He1_e_ex = self.R_He1_p_ex = self.R_He1_H_ex = 1e-16

        #                                     #electron           #ion                #cloud CX     #cloud ionization
        # self.rate_coeff_matrix = np.array([[self.R_He_e_ion, self.R_He_p_loss,   0,               self.R_He_H_ion],  #He
        #                                   [self.R_He1_e_ion, self.R_He1_p_loss, self.R_He1_H_cx, self.R_He1_H_ion],  #He1
        #                                   [0,                0,                 self.R_He2_H_cx, 0               ]]) #He2
        

        # electron_matrix = np.array([[self.R_He_e_ion/5,  self.R_He_e_ion/5,  self.R_He_e_ion/5,  self.R_He_e_ion/5, self.R_He_e_ion/5],
        #                             [self.R_He1_e_ion/5, self.R_He1_e_ex,    self.R_He1_e_ex,    self.R_He1_e_ex,   self.R_He1_e_ex],
        #                             [self.R_He1_e_ex,    self.R_He1_e_ion/5, self.R_He1_e_ex,    self.R_He1_e_ex,   self.R_He1_e_ex],
        #                             [self.R_He1_e_ex,    self.R_He1_e_ex,    self.R_He1_e_ion/5, self.R_He1_e_ex,   self.R_He1_e_ex],
        #                             [self.R_He1_e_ex,    self.R_He1_e_ex,    self.R_He1_e_ex,    self.R_He1_e_ion/5, self.R_He1_e_ex],
        #                             [self.R_He1_e_ex,    self.R_He1_e_ex,    self.R_He1_e_ex,    self.R_He1_e_ex,   self.R_He1_e_ion/5,]
        #                             [0.0,                0.0,                0.0,                0.0,               0.0]])


    def assemble_rate_coefficient_matrix(self):

        projectiles = ['He', 'He1', 'He1', 'He1', 'He1', 'He1', 'He2']
        states = [None, 1, 2, 3, 4, 5, None]
        #targets = ['e', 'p', 'Hi', 'Hd']

        self.electron_matrix = self.get_rate_coeff_matrix_for_target(projectiles, states, 'e')
        self.proton_matrix = self.get_rate_coeff_matrix_for_target(projectiles, states, 'p')
        self.Hi_matrix = self.get_rate_coeff_matrix_for_target(projectiles, states, 'Hi')
        self.Hd_matrix = self.get_rate_coeff_matrix_for_target(projectiles, states, 'Hd')

        self.rate_coeff_matrix = np.hstack((self.electron_matrix, self.proton_matrix, self.Hi_matrix, self.Hd_matrix))

    def get_rate_coeff_matrix_for_target(self, projectiles, states, target):
        matrix = np.zeros((7, 5), dtype=float)
        for i,proj in enumerate(projectiles):
            for j in range(5):
                matrix[i,j] = self.select_rate_coefficient(He_ion=proj, He_state=states[i], target=target, target_ind=j+1)
        return matrix

    def select_rate_coefficient(self, He_ion, He_state, target, target_ind):

        if He_ion == 'He' and not He_state:
            if target == 'e':
                return self.R_He_e_ion/5
            elif target == 'p':
                return self.R_He_p_loss/5
            elif target == 'Hi':
                return self.R_He_H_ion/5
            else:
                return 0.0
            
        elif He_ion == 'He1' and He_state:
            if target == 'e':
                if target_ind == He_state:
                    return self.R_He1_e_ion
                else:
                    return self.R_He1_e_ex
            elif target == 'p':
                if target_ind == He_state:
                    return self.R_He1_p_loss
                else:
                    return self.R_He1_p_ex
            elif target == 'Hi':
                if target_ind == He_state:
                    return self.R_He1_H_ion
                else:
                    return self.R_He1_H_ex
            elif target == 'Hd':
                if target_ind == 1:
                    return self.R_He1_H_cx
                else:
                    return 0.0
                
        elif He_ion == 'He2' and not He_state:
            if target == 'Hd':
                return self.R_He2_H_cx/5
            else:
                return 0.0

    def get_reaction_masks(self):
        nmax = 5
        self.reaction_masks = [self.assemble_mask(He_charge=0, nmax=nmax),
                               self.assemble_mask(He_charge=1, nmax=nmax, nstate=1),
                               self.assemble_mask(He_charge=1, nmax=nmax, nstate=2),
                               self.assemble_mask(He_charge=1, nmax=nmax, nstate=3),
                               self.assemble_mask(He_charge=1, nmax=nmax, nstate=4),
                               self.assemble_mask(He_charge=1, nmax=nmax, nstate=5),
                               self.assemble_mask(He_charge=2, nmax=nmax)]

    def assemble_mask(self, He_charge, nmax, nstate = None):
        if He_charge == 0 and not nstate:
            ion = He_ion_mask(nmax)
            cx = He_cx_mask(nmax)
        elif He_charge == 2 and not nstate:
            ion = He2_ion_mask(nmax)
            cx = He2_cx_mask(nmax)
        elif He_charge == 1 and nstate:
            ion = He1_i_ion_ex_mask(nmax, nstate)
            cx = He1_i_cx_mask(nmax, nstate)
        mask = np.hstack((ion, ion, ion, cx))
        return mask
        
def He_ion_mask(n):
    mask = np.zeros((n+2, n), dtype=float)
    mask[0] = -1
    return mask

def He1_i_ion_ex_mask(n, i):
    mask = np.zeros((n+2, n), dtype=float)
    mask[:-1,i-1] = 1
    mask[i] = -1
    return mask

def He2_ion_mask(n):
    mask = np.zeros((n+2, n), dtype=float)
    r = np.arange(n)
    mask[r+1,r] = 1
    return mask

def He_cx_mask(n):
    mask = np.zeros((n+2, n), dtype=float)
    mask[1:-1,0] = 1
    return mask

def He1_i_cx_mask(n,i):
    mask = np.zeros((n+2, n), dtype=float)
    mask[i,0] = -1
    mask[-1,i-1] = 1
    return mask

def He2_cx_mask(n):
    mask = np.zeros((n+2, n), dtype=float)
    mask[-1] = -1
    return mask