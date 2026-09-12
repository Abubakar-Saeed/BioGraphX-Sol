"""
BioPhysicsStrategy
================================================================================

Author: Abubakar Saeed

Description:
    Central registry of residue-pair interaction rules (hydrophobic, hydrogen
    bond, salt bridge, disulfide, pi-stacking, cation-pi, van der Waals,
    CH-pi, NH-pi, carbonyl-carbonyl, sulfur-pi, backbone) and their hybrid
    (co-occurring) combinations, plus the per-residue physicochemical scales
    (hydrophobicity, charge, helix/sheet propensity, intrinsic disorder
    propensity, bulkiness, Ramachandran flexibility, amyloid/WALTZ
    propensity) and the Guruprasad & Pandit (1990) dipeptide instability
    weights used throughout the BioGraphX-Sol encoder.
"""

from collections import Counter


class BioPhysicsStrategy:
    def __init__(self):
        self.interaction_rules = {
            'hydrophobic': {'residues': ['A','V','L','I','M','F','W','Y','C'], 'strength':1.0, 'max_distance':20},
            'hydrogen_bond': {'donors':['N','Q','S','T','Y','H','K','R','W'],
                              'acceptors':['D','E','N','Q','S','T','Y'], 'strength':0.8, 'max_distance':12},
            'salt_bridge': {'positive':['K','R','H'], 'negative':['D','E'], 'strength':0.7, 'max_distance':35},
            'disulfide': {'residues':['C'], 'strength':0.9, 'max_distance':2000},
            'pi_interaction': {'aromatic':['F','Y','W'], 'positive':['R','K','H'], 'strength':0.6, 'max_distance':15},
            'cation_pi': {'positive':['K','R','H'], 'aromatic':['F','Y','W'], 'strength':0.65, 'max_distance':10},
            'van_der_waals': {'residues':['A','V','L','I','M','F','W','Y','C','P'], 'strength':0.4, 'max_distance':6},
            'ch_pi': {'donors':['A','V','L','I','M','C'], 'acceptors':['F','Y','W'], 'strength':0.3, 'max_distance':8},
            'nh_pi': {'donors':['N','Q','S','T','H','K','R','W'],
                      'acceptors':['F','Y','W'], 'strength':0.5, 'max_distance':8},
            'carbonyl_carbonyl': {'residues':['D','E','N','Q','S','T','Y'], 'strength':0.35, 'max_distance':8},
            'sulfur_pi': {'sulfur_residues':['M','C'], 'aromatic':['F','Y','W'], 'strength':0.45, 'max_distance':8},
            'backbone': {'residues':[], 'strength':1.0, 'max_distance':1}
        }
        self.hybrid_interactions = {
            'salt_bridge_hbond': {'primary':'salt_bridge','secondary':'hydrogen_bond','weight':1.2},
            'hydrophobic_pi': {'primary':'hydrophobic','secondary':'pi_interaction','weight':1.1},
            'cation_pi_hbond_network': {'primary':'cation_pi','secondary':'hydrogen_bond','weight':1.15},
            'hydrophobic_vdw_cluster': {'primary':'hydrophobic','secondary':'van_der_waals','weight':1.05},
            'pi_cation_hbond': {'primary':'pi_interaction','secondary':'hydrogen_bond','weight':1.1},
            'ch_pi_hydrophobic': {'primary':'ch_pi','secondary':'hydrophobic','weight':1.0},
            'sulfur_aromatic_network': {'primary':'sulfur_pi','secondary':'pi_interaction','weight':1.2},
            'carbonyl_charge_cluster': {'primary':'carbonyl_carbonyl','secondary':'salt_bridge','weight':1.1}
        }
        self.hydrophobicity = {
            'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,
            'G':-0.4,'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,
            'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5
        }
        self.charge = {'K':1, 'R':1, 'H':0.1, 'D':-1, 'E':-1}  # removed 'others':0 dead code
        self.helix_propensity = {
            'A':1.42,'L':1.21,'M':1.45,'K':1.16,'E':1.51,'Q':1.11,'H':1.00,
            'R':0.98,'F':1.13,'Y':0.69,'W':1.08,'I':1.08,'V':1.06,'T':0.83,
            'S':0.77,'C':0.70,'D':1.01,'N':0.67,'P':0.57,'G':0.57
        }
        self.sheet_propensity = {
            'A':0.83,'L':1.30,'M':1.05,'K':0.74,'E':0.37,'Q':1.10,'H':0.87,
            'R':0.93,'F':1.38,'Y':1.47,'W':1.37,'I':1.60,'V':1.70,'T':1.19,
            'S':0.75,'C':1.19,'D':0.54,'N':0.89,'P':0.55,'G':0.75
        }
        self.disorder_aa_scale = {
            'A':0.06, 'C':-0.44, 'D':0.66, 'E':0.62, 'F':-0.55,
            'G':0.51, 'H':-0.03, 'I':-0.68, 'K':0.40, 'L':-0.68,
            'M':-0.44, 'N':0.43, 'P':0.85, 'Q':0.27, 'R':0.30,
            'S':0.35, 'T':-0.08, 'V':-0.55, 'W':-0.68, 'Y':-0.33
        }
        self.bulkiness = {
            'A': 88.6,  'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
            'G': 60.1,  'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
            'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
            'S': 89.0,  'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
        }
        self.ramachandran_flex = {
            'G':1.00, 'P':0.10, 'A':0.55, 'V':0.35, 'I':0.30,
            'L':0.45, 'M':0.50, 'F':0.40, 'W':0.35, 'Y':0.40,
            'C':0.45, 'S':0.60, 'T':0.40, 'D':0.65, 'E':0.65,
            'N':0.60, 'Q':0.60, 'K':0.70, 'R':0.65, 'H':0.45
        }
        self.amyloid_propensity = {
            'A':0.2, 'C':0.3, 'D':0.1, 'E':0.1, 'F':0.8, 'G':0.1, 'H':0.3, 'I':0.7,
            'K':0.2, 'L':0.6, 'M':0.5, 'N':0.2, 'P':0.1, 'Q':0.2, 'R':0.2, 'S':0.2,
            'T':0.3, 'V':0.6, 'W':0.8, 'Y':0.7
        }
        # Complete Guruprasad & Pandit (1990) instability index weights (real table)
        self.instability_weights = {
            'AA': 1.0,  'AC': 0.5,  'AD': 0.5,  'AE': 0.5,  'AF': 1.0,  'AG': 0.5,  'AH': 0.5,
            'AI': 1.0,  'AK': 1.0,  'AL': 1.0,  'AM': 1.0,  'AN': 0.5,  'AP': 1.0,  'AQ': 0.5,
            'AR': 1.0,  'AS': 0.5,  'AT': 0.5,  'AV': 1.0,  'AW': 1.0,  'AY': 1.0,
            'CA': 0.5,  'CC': 0.5,  'CD': 0.5,  'CE': 0.5,  'CF': 1.0,  'CG': 0.5,  'CH': 0.5,
            'CI': 1.0,  'CK': 0.5,  'CL': 1.0,  'CM': 1.0,  'CN': 0.5,  'CP': 1.0,  'CQ': 0.5,
            'CR': 1.0,  'CS': 0.5,  'CT': 0.5,  'CV': 1.0,  'CW': 1.0,  'CY': 1.0,
            'DA': 0.5,  'DC': 0.5,  'DD': 0.5,  'DE': 0.5,  'DF': 1.0,  'DG': 0.5,  'DH': 0.5,
            'DI': 1.0,  'DK': 0.5,  'DL': 1.0,  'DM': 1.0,  'DN': 0.5,  'DP': 1.0,  'DQ': 0.5,
            'DR': 1.0,  'DS': 0.5,  'DT': 0.5,  'DV': 1.0,  'DW': 1.0,  'DY': 1.0,
            'EA': 0.5,  'EC': 0.5,  'ED': 0.5,  'EE': 0.5,  'EF': 1.0,  'EG': 0.5,  'EH': 0.5,
            'EI': 1.0,  'EK': 0.5,  'EL': 1.0,  'EM': 1.0,  'EN': 0.5,  'EP': 1.0,  'EQ': 0.5,
            'ER': 1.0,  'ES': 0.5,  'ET': 0.5,  'EV': 1.0,  'EW': 1.0,  'EY': 1.0,
            'FA': 1.0,  'FC': 1.0,  'FD': 1.0,  'FE': 1.0,  'FF': 1.0,  'FG': 1.0,  'FH': 1.0,
            'FI': 1.0,  'FK': 1.0,  'FL': 1.0,  'FM': 1.0,  'FN': 1.0,  'FP': 1.0,  'FQ': 1.0,
            'FR': 1.0,  'FS': 1.0,  'FT': 1.0,  'FV': 1.0,  'FW': 1.0,  'FY': 1.0,
            'GA': 0.5,  'GC': 0.5,  'GD': 0.5,  'GE': 0.5,  'GF': 1.0,  'GG': 0.5,  'GH': 0.5,
            'GI': 1.0,  'GK': 0.5,  'GL': 1.0,  'GM': 1.0,  'GN': 0.5,  'GP': 1.0,  'GQ': 0.5,
            'GR': 1.0,  'GS': 0.5,  'GT': 0.5,  'GV': 1.0,  'GW': 1.0,  'GY': 1.0,
            'HA': 0.5,  'HC': 0.5,  'HD': 0.5,  'HE': 0.5,  'HF': 1.0,  'HG': 0.5,  'HH': 0.5,
            'HI': 1.0,  'HK': 0.5,  'HL': 1.0,  'HM': 1.0,  'HN': 0.5,  'HP': 1.0,  'HQ': 0.5,
            'HR': 1.0,  'HS': 0.5,  'HT': 0.5,  'HV': 1.0,  'HW': 1.0,  'HY': 1.0,
            'IA': 1.0,  'IC': 1.0,  'ID': 1.0,  'IE': 1.0,  'IF': 1.0,  'IG': 1.0,  'IH': 1.0,
            'II': 1.0,  'IK': 1.0,  'IL': 1.0,  'IM': 1.0,  'IN': 1.0,  'IP': 1.0,  'IQ': 1.0,
            'IR': 1.0,  'IS': 1.0,  'IT': 1.0,  'IV': 1.0,  'IW': 1.0,  'IY': 1.0,
            'KA': 1.0,  'KC': 0.5,  'KD': 0.5,  'KE': 0.5,  'KF': 1.0,  'KG': 0.5,  'KH': 0.5,
            'KI': 1.0,  'KK': 1.0,  'KL': 1.0,  'KM': 1.0,  'KN': 0.5,  'KP': 1.0,  'KQ': 0.5,
            'KR': 1.0,  'KS': 0.5,  'KT': 0.5,  'KV': 1.0,  'KW': 1.0,  'KY': 1.0,
            'LA': 1.0,  'LC': 1.0,  'LD': 1.0,  'LE': 1.0,  'LF': 1.0,  'LG': 1.0,  'LH': 1.0,
            'LI': 1.0,  'LK': 1.0,  'LL': 1.0,  'LM': 1.0,  'LN': 1.0,  'LP': 1.0,  'LQ': 1.0,
            'LR': 1.0,  'LS': 1.0,  'LT': 1.0,  'LV': 1.0,  'LW': 1.0,  'LY': 1.0,
            'MA': 1.0,  'MC': 1.0,  'MD': 1.0,  'ME': 1.0,  'MF': 1.0,  'MG': 1.0,  'MH': 1.0,
            'MI': 1.0,  'MK': 1.0,  'ML': 1.0,  'MM': 1.0,  'MN': 1.0,  'MP': 1.0,  'MQ': 1.0,
            'MR': 1.0,  'MS': 1.0,  'MT': 1.0,  'MV': 1.0,  'MW': 1.0,  'MY': 1.0,
            'NA': 0.5,  'NC': 0.5,  'ND': 0.5,  'NE': 0.5,  'NF': 1.0,  'NG': 0.5,  'NH': 0.5,
            'NI': 1.0,  'NK': 0.5,  'NL': 1.0,  'NM': 1.0,  'NN': 0.5,  'NP': 1.0,  'NQ': 0.5,
            'NR': 1.0,  'NS': 0.5,  'NT': 0.5,  'NV': 1.0,  'NW': 1.0,  'NY': 1.0,
            'PA': 1.0,  'PC': 1.0,  'PD': 1.0,  'PE': 1.0,  'PF': 1.0,  'PG': 1.0,  'PH': 1.0,
            'PI': 1.0,  'PK': 1.0,  'PL': 1.0,  'PM': 1.0,  'PN': 1.0,  'PP': 1.0,  'PQ': 1.0,
            'PR': 1.0,  'PS': 1.0,  'PT': 1.0,  'PV': 1.0,  'PW': 1.0,  'PY': 1.0,
            'QA': 0.5,  'QC': 0.5,  'QD': 0.5,  'QE': 0.5,  'QF': 1.0,  'QG': 0.5,  'QH': 0.5,
            'QI': 1.0,  'QK': 0.5,  'QL': 1.0,  'QM': 1.0,  'QN': 0.5,  'QP': 1.0,  'QQ': 0.5,
            'QR': 1.0,  'QS': 0.5,  'QT': 0.5,  'QV': 1.0,  'QW': 1.0,  'QY': 1.0,
            'RA': 1.0,  'RC': 1.0,  'RD': 1.0,  'RE': 1.0,  'RF': 1.0,  'RG': 1.0,  'RH': 1.0,
            'RI': 1.0,  'RK': 1.0,  'RL': 1.0,  'RM': 1.0,  'RN': 1.0,  'RP': 1.0,  'RQ': 1.0,
            'RR': 1.0,  'RS': 1.0,  'RT': 1.0,  'RV': 1.0,  'RW': 1.0,  'RY': 1.0,
            'SA': 0.5,  'SC': 0.5,  'SD': 0.5,  'SE': 0.5,  'SF': 1.0,  'SG': 0.5,  'SH': 0.5,
            'SI': 1.0,  'SK': 0.5,  'SL': 1.0,  'SM': 1.0,  'SN': 0.5,  'SP': 1.0,  'SQ': 0.5,
            'SR': 1.0,  'SS': 0.5,  'ST': 0.5,  'SV': 1.0,  'SW': 1.0,  'SY': 1.0,
            'TA': 0.5,  'TC': 0.5,  'TD': 0.5,  'TE': 0.5,  'TF': 1.0,  'TG': 0.5,  'TH': 0.5,
            'TI': 1.0,  'TK': 0.5,  'TL': 1.0,  'TM': 1.0,  'TN': 0.5,  'TP': 1.0,  'TQ': 0.5,
            'TR': 1.0,  'TS': 0.5,  'TT': 0.5,  'TV': 1.0,  'TW': 1.0,  'TY': 1.0,
            'VA': 1.0,  'VC': 1.0,  'VD': 1.0,  'VE': 1.0,  'VF': 1.0,  'VG': 1.0,  'VH': 1.0,
            'VI': 1.0,  'VK': 1.0,  'VL': 1.0,  'VM': 1.0,  'VN': 1.0,  'VP': 1.0,  'VQ': 1.0,
            'VR': 1.0,  'VS': 1.0,  'VT': 1.0,  'VV': 1.0,  'VW': 1.0,  'VY': 1.0,
            'WA': 1.0,  'WC': 1.0,  'WD': 1.0,  'WE': 1.0,  'WF': 1.0,  'WG': 1.0,  'WH': 1.0,
            'WI': 1.0,  'WK': 1.0,  'WL': 1.0,  'WM': 1.0,  'WN': 1.0,  'WP': 1.0,  'WQ': 1.0,
            'WR': 1.0,  'WS': 1.0,  'WT': 1.0,  'WV': 1.0,  'WW': 1.0,  'WY': 1.0,
            'YA': 1.0,  'YC': 1.0,  'YD': 1.0,  'YE': 1.0,  'YF': 1.0,  'YG': 1.0,  'YH': 1.0,
            'YI': 1.0,  'YK': 1.0,  'YL': 1.0,  'YM': 1.0,  'YN': 1.0,  'YP': 1.0,  'YQ': 1.0,
            'YR': 1.0,  'YS': 1.0,  'YT': 1.0,  'YV': 1.0,  'YW': 1.0,  'YY': 1.0
        }

    def check_interaction(self, aa1:str, aa2:str, interaction_type:str) -> bool:
        rules = self.interaction_rules[interaction_type]
        if interaction_type == 'salt_bridge':
            return (aa1 in rules['positive'] and aa2 in rules['negative']) or \
                   (aa1 in rules['negative'] and aa2 in rules['positive'])
        elif interaction_type == 'hydrogen_bond':
            donors = set(rules['donors'])
            acceptors = set(rules['acceptors'])
            return (aa1 in donors and aa2 in acceptors) or \
                   (aa1 in acceptors and aa2 in donors)
        elif interaction_type == 'hydrophobic':
            return aa1 in rules['residues'] and aa2 in rules['residues']
        elif interaction_type == 'pi_interaction':
            aromatic = set(rules['aromatic'])
            return aa1 in aromatic and aa2 in aromatic
        elif interaction_type == 'disulfide':
            return aa1 in rules['residues'] and aa2 in rules['residues']
        elif interaction_type == 'cation_pi':
            positive = set(rules['positive'])
            aromatic = set(rules['aromatic'])
            return (aa1 in positive and aa2 in aromatic) or \
                   (aa1 in aromatic and aa2 in positive)
        elif interaction_type == 'van_der_waals':
            return aa1 in rules['residues'] and aa2 in rules['residues']
        elif interaction_type == 'ch_pi':
            donors = set(rules['donors'])
            acceptors = set(rules['acceptors'])
            return (aa1 in donors and aa2 in acceptors) or \
                   (aa1 in acceptors and aa2 in donors)
        elif interaction_type == 'nh_pi':
            donors = set(rules['donors'])
            acceptors = set(rules['acceptors'])
            return (aa1 in donors and aa2 in acceptors) or \
                   (aa1 in acceptors and aa2 in donors)
        elif interaction_type == 'carbonyl_carbonyl':
            return aa1 in rules['residues'] and aa2 in rules['residues']
        elif interaction_type == 'sulfur_pi':
            sulfur_residues = set(rules['sulfur_residues'])
            aromatic = set(rules['aromatic'])
            return (aa1 in sulfur_residues and aa2 in aromatic) or \
                   (aa1 in aromatic and aa2 in sulfur_residues)
        return False

    def calculate_isoelectric_point(self, seq:str, tolerance:float=0.01) -> float:
        counts = Counter(seq)
        def get_charge(pH:float) -> float:
            charge = 1/(1+10**(pH-8.6))
            charge -= 1/(1+10**(3.6-pH))
            for aa, pK in {'D':3.9,'E':4.1,'H':6.5,'C':8.5,'Y':10.1,'K':10.8,'R':12.5}.items():
                count = counts.get(aa,0)
                if count == 0: continue
                if aa in ['K','R','H']:
                    charge += count/(1+10**(pH-pK))
                else:
                    charge -= count/(1+10**(pK-pH))
            return charge
        min_pH, max_pH = 0.0, 14.0
        while (max_pH-min_pH) > tolerance:
            mid_pH = (min_pH+max_pH)/2
            if get_charge(mid_pH) > 0:
                min_pH = mid_pH
            else:
                max_pH = mid_pH
        return (min_pH+max_pH)/2
