"""
Dictionary of NMR isotopes with their spin quantum numbers and gyromagnetic ratios in MHz/T.

From: https://www.kherb.io/docs/nmr_table.html
"""

ISOTOPES =\
{'1H': [0.5, 42.577478615342585],
 '2D': [1.0, 6.5359028540009305],
 '3T': [0.5, 45.41483815473964],
 '3He': [0.5, -32.43604456417949],
 '6Li': [1.0, 6.266099405837534],
 '7Li': [1.5, 16.548177299618295],
 '9Be': [1.5, -5.983379963834242],
 '10B': [3.0, 4.57473388220653],
 '11B': [1.5, 13.66160796005943],
 '13C': [0.5, 10.707746367473971],
 '14N': [1.0, 3.076272817251739],
 '15N': [0.5, -4.3152552187859134],
 '17O': [2.5, -5.7734832203316975],
 '19F': [0.5, 40.06924371705693],
 '21Ne': [1.5, -3.362579959801532],
 '22Na': [3.0, 4.436349259342206],
 '23Na': [1.5, 11.268733657034751],
 '25Mg': [2.5, -2.607933066661972],
 '26Al': [5.0, 4.273225764239245],
 '27Al': [2.5, 11.100630067688776],
 '29Si': [0.5, -8.461871234008282],
 '31P': [0.5, 17.241162495263175],
 '33S': [1.5, 3.2688220630834754],
 '35Cl': [1.5, 4.175656570906633],
 '37Cl': [1.5, 3.4759025124743057],
 '39Ar': [3.5, -3.462835209795831],
 '39K': [1.5, 1.9893443809332112],
 '41K': [1.5, 1.091921234883595],
 '43Ca': [3.5, -2.868991639572541],
 '45Sc': [3.5, 10.35365948891156],
 '53Mn': [3.5, 10.961289063460638],
 '60Co': [5.0, 5.7916463354780205],
 '63Ni': [0.5, 7.561612483277437],
 '69Ga': [1.5, 10.23978520568125],
 '71Ga': [1.5, 13.010902748192017],
 '73Ge': [4.5, -1.4876591727852992],
 '77Se': [0.5, 8.134221686648205],
 '79Se': [3.5, -2.21708568778123],
 '81Kr': [3.5, -1.9753405882294452],
 '83Kr': [4.5, -1.6443288722876133],
 '85Kr': [4.5, -1.703226109304539],
 '85Rb': [2.5, 4.125530397832004],
 '87Rb': [1.5, 13.981309683545954],
 '87Sr': [4.5, -1.851714225407608],
 '89Y': [0.5, -2.0931336103407774],
 '91Nb': [4.5, 11.032433266932337],
 '92Nb': [7.0, 5.58627189504799],
 '97Tc': [4.5, 9.858553909649345],
 '99Ru': [2.5, -1.954432903943886],
 '101Ru': [2.5, -2.189208775400484],
 '102Rh': [6.0, 5.094433141455396],
 '108Ag': [6.0, 4.548147293369158],
 '111Cd': [0.5, -9.05564075618306],
 '113Cd': [5.5, -1.5083033111346835],
 '113In': [4.5, 9.351736155393834],
 '115In': [4.5, 9.371724288750167],
 '121Sn': [5.5, -1.920754901131635],
 '125Sb': [3.5, 5.727834340731468],
 '129Xe': [0.5, -11.860160502223788],
 '131Xe': [1.5, 3.5157686750625525],
 '133Cs': [3.5, 5.614148807428737],
 '134Cs': [4.0, 5.69655448494487],
 '135Cs': [3.5, 5.941920316280481],
 '137Cs': [3.5, 6.179527436650749],
 '133Ba': [0.5, 11.767759427100511],
 '135Ba': [1.5, 4.258996923544905],
 '137Ba': [1.5, 4.7641207681939495],
 '137La': [3.5, 5.871574670194692],
 '138La': [5.0, 5.653524946166542],
 '139La': [3.5, 6.052556812291568],
 '141Pr': [2.5, 13.00719308615385],
 '143Nd': [3.5, -2.319446225429283],
 '145Nd': [3.5, -1.4286917595132482],
 '145Pm': [2.5, 11.586341708247684],
 '147Pm': [3.5, 5.618940151744178],
 '147Sm': [3.5, -1.7619079778143567],
 '149Sm': [3.5, -1.454172999736274],
 '151Sm': [2.5, -1.099177943637708],
 '150Eu': [5.0, 4.119249381011216],
 '151Eu': [2.5, 10.560340659609436],
 '152Eu': [3.0, -4.917588978540037],
 '153Eu': [2.5, 4.66319763384053],
 '154Eu': [3.0, -5.081728819406879],
 '155Eu': [2.5, 4.622340534132499],
 '155Gd': [1.5, -1.3166759371083223],
 '157Gd': [1.5, -1.726771452834457],
 '157Tb': [1.5, 10.315909503395964],
 '158Tb': [3.0, 4.4541353102101295],
 '159Tb': [1.5, 10.20919319818842],
 '161Dy': [2.5, -1.4604888626975372],
 '163Dy': [2.5, 2.04590402269321],
 '163Ho': [3.5, 9.190669550527298],
 '165Ho': [3.5, 9.059996523742551],
 '166Ho': [7.0, 3.941969641339908],
 '167Er': [3.5, -1.2246240493510665],
 '169Tm': [0.5, -3.5216380718489675],
 '171Tm': [0.5, -3.506392885390747],
 '171Yb': [0.5, 7.505205293382021],
 '173Yb': [2.5, -2.067247283734719],
 '173Lu': [3.5, 4.950329831362216],
 '174Lu': [1.0, 15.107979780096652],
 '175Lu': [3.5, 4.847315928580239],
 '176Lu': [7.0, 3.4410563719983727],
 '177Hf': [3.5, 1.7227060697789325],
 '178Hf': [16.0, 3.8732301845416814],
 '179Hf': [4.5, -1.0822388475730185],
 '179Ta': [3.5, 4.978642320498912],
 '180Ta': [9.0, 4.077240422770786],
 '191Ir': [1.5, 0.7632756686749131],
 '193Ir': [1.5, 0.8283217975633214],
 '193Pt': [0.5, 9.162357061390605],
 '197Au': [1.5, 0.7378670245778789],
 '199Hg': [0.5, 7.68204945629738],
 '201Hg': [1.5, -2.8356046812290385],
 '204Tl': [2.0, 0.3430166953099643],
 '205Pb': [2.5, 2.162987054692344],
 '207Pb': [0.5, 9.00380712222511],
 '207Bi': [4.5, 6.899124436187426],
 '208Bi': [5.0, 6.970099248698476],
 '210Bi': [9.0, 2.303717064797785],
 '209Po': [0.5, 10.366726791590036],
 '229Th': [2.5, 1.402557154156299],
 '231Pa': [1.5, 10.11264035061969],
 '233U': [2.5, -1.7989320020700357],
 '235U': [3.5, -0.827595836303406],
 '239Pu': [0.5, 3.079527664560569],
 '241Pu': [2.5, -2.067247283734719],
 '241Am': [2.5, 4.878459666630606],
 '242Am': [5.0, 1.524518645822064],
 '243Am': [2.5, 4.634536683299075],
 '243Cm': [2.5, 1.2196149166576515],
 '245Cm': [3.5, 1.0889418898729026],
 '247Cm': [4.5, 0.6098074583288254]}