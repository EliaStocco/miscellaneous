#!/bin/bash --login

source ~/.elia

CALC="scf"

source var.sh

OUT_DIR="./outdir"
RES_DIR="./results"

for DIR in ${OUT_DIR} ${RES_DIR} ; do
    if test ! -d ${DIR} ; then
        mkdir ${DIR}
    fi
done

############################################################################
restart_mode="from_scratch"
startingpot="atomic"
startingwfc="atomic"

#restart_mode="restart"
# startingpot="file"
# startingwfc="file"

occupations="fixed"
#occupations="smearing"

xc="lda"
pseudo_dir="${PSEUDO_DIR}/${xc}"

ratio="12"
ecutwfc="80"
ecutrho=$((ecutwfc*ratio))
electron_maxstep="80"
conv_thr="1.0D-8"
max_seconds="$((2*3600-400))"
# nk="2"

e_thr="1.0D-7"
f_thr="1.0D-6"

# c_bands eigenvalue not converged
# - increase k points
# - increase cutoff
# - change mixing_mode
# - lower mixing_beta (or higher?)
# - select 'cg' style diagonalization
# - modify 'mixing_ndim'

############################################################################
# read input arguments
source ~/read.sh $@

###########################################################################
short_name="$PREFIX.nk=${nk}"
full_name="$short_name"
OUT_DIR_CYCLE="$OUT_DIR/$short_name"
#yn="!"

############################################################################
cat > ${INPUT_FILE} << EOF
&CONTROL
  calculation = '$CALC'
  restart_mode= '${restart_mode}'
  prefix      = '${PREFIX}'
  pseudo_dir  = '${pseudo_dir}'
  outdir      = '${OUT_DIR_CYCLE}'
  tprnfor     = .true.
  !tstress     = .true.
  verbosity   = 'low'
  lelpol      = .true.
  !max_seconds = ${max_seconds}
  !etot_conv_thr = ${e_thr}
  !forc_conv_thr = ${f_thr}
  !nstep = 100
/
&SYSTEM
  ecutwfc = ${ecutwfc}
  ecutrho = ${ecutrho}
  ibrav = 0
  nat   = 30
  ntyp  = 3
  occupations = 'fixed'
/
&ELECTRONS
  !mixing_mode = 'local-TF'
  !mixing_beta = 0.9
  !mixing_ndim=25
  !diagonalization = 'davidson' !ppcg, paro
  diago_david_ndim = 4
  conv_thr = ${conv_thr}
  ! diago_thr_init = 1.0D-4
  electron_maxstep=${electron_maxstep}
  startingpot = '${startingpot}'
  startingwfc = '${startingwfc}'
  reset_ethr = .true.
/
&IONS
  !upscale = 10
  pot_extrapolation = 'second_order'
  wfc_extrapolation = 'second_order'
  !trust_radius_min = 1.0D-4
/
&CELL
  cell_dofree = 'all'
/
ATOMIC_SPECIES
  Nb  92.90638 $(basename $(ls ${pseudo_dir}/Nb.*))
  Li  6.941    $(basename $(ls ${pseudo_dir}/Li.*))
  O   15.9994  $(basename $(ls ${pseudo_dir}/O.*))

K_POINTS automatic
  ${nk} ${nk} ${nk} 0 0 0

CELL_PARAMETERS angstrom
  5.0747482210040511	  0.0000000000000000	  0.0000000000000000
 -2.5373733908355391	  4.3948578270373035	  0.0000000000000000
  0.0000058698089519	 -0.0000029162089701	 13.6997931761245120

ATOMIC_POSITIONS angstrom
Nb      0.000000494181352      0.000009096409500      0.014536468762931
Nb     -0.000001169404431      0.000002345524929      6.864438794751362
Nb      2.537370401259083      1.464946817476477      4.581141307767273
Nb      2.537382359048801      1.464961031963810     11.431034299156847
Nb      0.000002247295183      2.929913729184866      9.147729850743236
Nb     -0.000000708693964      2.929904262818772      2.297840563525049
Li     -0.000007257850367     -0.000006043271016      2.984504379757907
Li      0.000004618165192      0.000008668633290      9.834327188045563
Li      2.537367233133835      1.464966849235031      7.551043705615621
Li      2.537374093893501      1.464946199232927      0.701193386532200
Li      0.000004679361718      2.929906918343773     12.117606561093918
Li      0.000003366464945      2.929897577933307      5.267805192573498
O      -0.661228616250037      1.407080381321495      1.410756787745298
O       1.649412740271903      3.118670813615414      1.410748703883498
O      -0.988182951900407      4.263952397038122      1.410749033456858
O       0.988188416679506      4.263968264690654      8.260632302146469
O       0.661229184621829      1.407091652377346      8.260629840459819
O      -1.649419640940630      3.118683004618097      8.260640711202562
O       1.876141955802612      2.872030359325684      5.977345938199521
O       1.649419557063024      0.188760200709408      5.977342654437217
O       4.086560986780540      1.334054360903575      5.977337833171543
O       0.988190683485276      1.334051250920438     12.827235919630780
O       3.198610396090979      2.872044228559225     12.827236411792377
O       3.425343625055183      0.188770676615707     12.827234129354542
O      -0.661217564866065      4.336999354727404     10.543931327558004
O      -0.887950972703138      1.653731191218414     10.543931762999236
O       1.549193448826867      2.799006582612163     10.543926318706086
O      -1.549196809560017      2.798994430562472      3.694039746103842
O       0.661227344261424      4.336979894544531      3.694050199591062
O       0.887952549822036      1.653712020396869      3.694048282738772

EOF

