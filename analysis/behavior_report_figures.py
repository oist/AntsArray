"""Readable schematic and figure descriptions for the behavioral report."""
import re
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def overview():
    fig,ax=plt.subplots(figsize=(15,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009',fc=color,ec='#34495e',lw=1.2))
        ax.text(x+w/2,y+h-.04,title,ha='center',va='top',weight='bold',fontsize=12)
        ax.text(x+w/2,y+h/2-.025,body,ha='center',va='center',fontsize=10,linespacing=1.5)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,lw=1.5,color='#34495e'))
    box(.015,.61,.215,.29,'TRACKING','ArUco identity + SLEAP\nBody and antenna landmarks\nPosition and exact contacts','#edf2f7')
    box(.265,.61,.215,.29,'STATE DEFINITION','Both colonies; >60% detected\nBody-aligned posture PCs\nUnsigned motion + contacts\nShort windows; Leiden','#e4f3ed')
    box(.515,.61,.215,.29,'BEHAVIORAL STATES','S0  ·  S1  ·  S2  ·  S3\nS4  ·  S5  ·  S6  ·  S7\nS0 = putative sleep state','#fff0d7')
    box(.765,.61,.215,.29,'ANT CLASSES','Right colony\nEach ant’s state proportions\nAll eight dimensions\nFour reproducible classes','#e9e5f6')
    for x in [.23,.48,.73]:arrow((x,.75),(x+.034,.75))
    box(.015,.13,.285,.29,'S0 → MOTION ENVELOPE','Body + antenna speeds during S0\nEqual-ant 95th percentiles\nFresh trailing-window motion labels','#fff0d7')
    box(.355,.13,.285,.29,'SLEEP AND RETURNS','S0 occupancy and quiet runs\nMotion-defined sleep over time\nReturn-contact responses by class','#e5f1fa')
    box(.695,.13,.285,.29,'CLASS ANALYSIS','Spatial occupancy emerges\nSpeed and sleep time courses\nTrip rate and time away','#e9e5f6')
    arrow((.60,.61),(.60,.51));arrow((.60,.51),(.16,.51));arrow((.16,.51),(.16,.42))
    arrow((.30,.275),(.355,.275));arrow((.87,.61),(.87,.42))
    arrow((.80,.60),(.63,.43))
    ax.text(.5,.045,'Spatial groups and historical sleep labels do not define states or ant classes.\n'
        'S0 remains one learned state; its motion envelope is a separate, broader readout for downstream sleep analysis.',
        ha='center',va='center',fontsize=11,color='#34495e')
    fig.suptitle('From tracked ants to behavioral states, ant classes, and downstream analysis',fontsize=16)
    fig.tight_layout();return fig


NOTES={
'00_workflow':('Analysis overview',
    'Identity tracking links each ant across time, while SLEAP provides body and antennal landmarks. '
    'Both colonies contribute posture, motion and exact contact features to the behavioral-state model. '
    'S0 is interpreted as the putative sleep state. Right-colony ants are grouped by their proportions of all eight states; '
    'spatial occupancy is examined only afterward. A separate branch calibrates a motion envelope from S0 and applies it '
    'to every track, retaining partial-pose observations for return-contact analyses. This broader motion label is not identical to S0 membership.'),
'01_full_body_pca':('Full-body posture coordinates',
    'All ten landmarks are translated, rotated and normalized by body-axis length before PCA over the full recording. '
    'The left panel shows cumulative explained variance; the right shows coordinate loadings for the first eight modes. '
    'Four PCs retain about 89.8% of the measured posture variance. The fit includes both colonies and excludes invalid full-body observations. '
    'This describes joint articulation without selecting antenna straightness as a starting feature.'),
'02_feature_expansion':('Choosing a reproducible behavioral vocabulary',
    'Each point is a feature representation and Leiden resolution. Seed agreement and agreement after removing ants measure reproducibility; '
    'outlined points pass all stated size and stability requirements. The selected eight-state solution uses four posture PCs, five motion features '
    'and two concurrent-contact features. More features or higher resolution produced finer partitions that failed one or more checks. '
    'Eight is the supported resolution under these tests, not an assertion that ants perform only eight actions.'),
'03_state_umaps':('Views of the same eight states',
    'Dots are sampled short behavioral windows, colored by their fixed Leiden state. Each panel changes only UMAP display settings. '
    'Leiden was fitted in the balanced feature space, not in these two-dimensional coordinates. Gaps and camera changes delimit windows. '
    'Visible island size, distance and separation depend on the embedding parameters and do not constitute independent evidence for additional states.'),
'04_state_profiles':('Features distinguishing behavioral states',
    'Rows are states and columns are the features used for clustering. Values summarize the ant-balanced dictionary and are shown after the '
    'same transformations and family scaling used for fitting. Color compares standardized features, not physical units. '
    'Posture, movement and concurrent contacts jointly define a state; no spatial labels or historical sleep labels entered the fit.'),
'04_mean_posture_motion':('Mean posture and motion of each state',
    'Skeletons show actual mean body-aligned coordinates for all assigned observations, with a common scale across states. Body segments are teal '
    'and antennae orange. The heatmap prints motion and contact measurements in physical units, with color scaled separately within each feature. '
    'Means are weighted by observed duration, so partial windows contribute their measured length. These pooled means can differ from the '
    'ant-balanced dictionary medians in the previous figure and can conceal variation within each state.'),
'04_quiet_states':('S0 as the putative sleep state',
    'Each point summarizes median unsigned body and antennal motion in one learned state. The outlined state, S0, is the user-selected putative '
    'sleep interpretation. Dotted reference limits are descriptive and were not used to learn the states. S1 is also quiet but remains a separate '
    'behavioral state. Low motion supports a putative sleep interpretation; state membership alone does not establish physiological sleep.'),
'05_ant_methods':('Selecting ant classes from state proportions',
    'Each ant is represented by square roots of its eight pooled state proportions, retaining every state dimension. Clustering methods and '
    'class counts are compared using ant-removal and temporal stability. Four classes are the finest partition passing the specified criteria, '
    'including a minimum class size and stability of the smallest class. A three-class partition is more temporally stable; the four-class result '
    'should be read as a supported refinement rather than a unique biological taxonomy.'),
'06_state_to_ant_classes':('From states to ant behavioral classes',
    'Each UMAP point is one ant, labeled by identity and colored by the fixed behavioral class. The heatmap gives the equal-ant mean state '
    'proportions within each class. The four classes contain 10, 14, 8 and 12 ants. Clustering used all eight square-root proportion dimensions; '
    'UMAP is a display only. Class differences in S0 occupancy are partly inherent in this construction, rather than independent validation.'),
'06_ant_umap_comparison':('Sensitivity of the ant-class display',
    'Nine neighborhood/minimum-distance settings display the same ant classes. Three random seeds were tested for each setting; panels show '
    'one fixed seed and report median trustworthiness and five-neighbor overlap. The selected setting, eight neighbors and minimum distance '
    '0.5, best preserves neighborhoods under this rule. Neither the class labels nor spatial groups enter the display-selection criterion. '
    'Embedding appearance does not change ant membership or select the number of classes.'),
'06_ant_classes':('Individual state proportions and observed coverage',
    'Rows are right-colony ants ordered by behavioral class and identity. The heatmap gives the fraction of each ant’s classified observations '
    'in each state. Bars compare tracking detection and usable state classification over the full recording. Full-body posture and motion '
    'requirements exclude observations that still have a tracked position. Missing time is excluded from the proportion denominator and '
    'never becomes an extra state; behavior-dependent missingness remains a possible source of bias.'),
'06_state_coverage':('Where classification coverage is lost',
    'Bars show equal-ant mean coverage after successive measurement requirements, separately for the two colonies. Complete landmarks and '
    'motion support account for most of the loss, followed by removal of segments shorter than 250 ms. Each stage uses its matching temporal '
    'sampling denominator, so sampling at 12 Hz does not artificially halve coverage. This audit motivates using S0-calibrated motion labels '
    'for downstream return responses while retaining the complete-posture requirement for state discovery.'),
'04_occupancy':[
    ('Behavioral classes and independent spatial groups','Cells show numbers of ants and color shows row-normalized overlap with frozen spatial groups. '
     'The selected four-class behavior partition is shown alongside the most stable passing two-class behavioral fit. Spatial labels enter only '
     'this comparison. Fine classes can subdivide a spatial group, reducing adjusted Rand agreement while preserving its broad organization. '
     'Agreement is descriptive because these comparisons use the same recording, rather than a held-out experiment.'),
    ('Spatial occupancy emerging from behavioral classes','Each map averages spatial occupancy for ants already assigned to a behavioral class. '
     'Behavioral classes were learned from state proportions without coordinates, nest occupancy or spatial group labels. The maps therefore '
     'show where the behavior-defined groups spend their observed time. Compare localization and spread across classes, while allowing for '
     'differences in position coverage and individual variation.'),
    ('Individual examples of spatial occupancy','Example ant occupancy maps are arranged by their behavioral class. They show how consistently '
     'the class-average spatial pattern appears in individual ants and where within-class differences remain. Examples illustrate observed '
     'variation rather than provide an independent test of class discovery. All class labels are fixed before these spatial maps are examined.'),
    ('Nest occupancy by behavioral class','Nest use is measured from annotated spatial regions after behavioral classification. The figure '
     'summarizes variation among ants in each class, providing a spatial description of the activity-defined groups. Nest occupancy did not '
     'choose the features, state number, class number or ant labels. Differences describe association between behavior and location, not a '
     'causal explanation for task allocation.')],
'05_speed':('Speed over time by behavioral class',
    'Each colored curve is the equal-ant class mean trajectory speed in ten-minute bins, using the grid-occupancy motion cache. Dark periods '
    'are shaded and the horizontal axis follows recording clock time. These are mean trajectory speeds, distinct from the within-window '
    'peak velocity features used for state discovery. Missing velocity is excluded rather than set to zero; the number of contributing ants '
    'can change with time.'),
'05_motion_calibration':('Calibrating motion thresholds from S0',
    'Cumulative distributions show the same body and antennal motion measurements later used for return analysis, evaluated at observations '
    'assigned to each state. Ants have equal weight within a state. Dashed lines are the 95th percentiles of the S0 distributions, giving a '
    'joint motion envelope containing at least 90% of sampled S0 observations. The companion panel reports overlap by state and transfer '
    'to held-out alternating 30-minute blocks. Overlap with S1 is expected: this is a motion readout informed by S0, not exact state recovery.'),
'05_motion_sleep':('Sleep over time: the counterpart to the speed figure',
    'Curves show the equal-ant mean percentage of classified time labeled as putative sleep, in ten-minute bins, using fresh S0-calibrated '
    'motion labels. Clock time and light shading match the speed plot. Labels require at least 50% jointly valid body/antenna observations '
    'in a trailing ten-second window, with at least 90% of those observations below both thresholds. Ant-bins also need 50% label coverage. '
    'The trailing window can delay transitions; unknown periods are excluded. This broader motion definition differs from direct S0 occupancy.'),
'05_quiet_bouts':('Observed S0 run lengths',
    'Cumulative curves describe contiguous runs of direct S0 membership, separately by ant class. Missing posture or a recording boundary '
    'censors a run; it is not counted as an observed awakening. The legends report the fraction ending at gaps. The curves characterize '
    'observed continuity and can underestimate uninterrupted quietness. This figure retains direct S0 membership, whereas the return figures '
    'use the separately calibrated motion labels.'),
'05_class_sleep_speed':('S0 occupancy and speed across ant classes',
    'Top panels show one point per ant, with bars marking equal-ant class means over the full recording. Bottom panels show ten-minute '
    'class time courses. Here putative sleep means direct S0 membership; unknown state intervals are excluded. Speed is measured from '
    'the trajectory cache. Because ant classes were fitted from state proportions, differences in total S0 occupancy describe their '
    'construction and are not independent evidence validating the classes.'),
'05_speed_sleep':[
    ('Direct S0 occupancy and time-dependent support','The left panel shows equal-ant mean direct S0 membership in ten-minute bins; the right '
     'shows how many ants meet the minimum classified-coverage requirement in each bin. Missing intervals are excluded rather than treated '
     'as waking. Changes in the contributing population can influence apparent trends, so interpret both panels together. These state labels '
     'are kept separate from the broader S0-calibrated motion labels used in the return analysis.'),
    ('Individual S0 occupancy through time','Each row is an ant ordered by behavioral class; color shows the fraction of classified '
     'observations in S0 within a ten-minute bin. Missing or insufficiently covered bins remain missing. The display shows heterogeneity '
     'and temporal organization hidden by class means. S0 is the sole putative sleep state here, with no relabeling of S1 or interpolation '
     'through missing full-body posture.')],
'06_clock_profiles':('Individual activity and S0 occupancy by time of day',
    'Matrices align each ant’s speed and direct S0 occupancy to the light cycle. Only complete light-on-to-light-on cycles are used when '
    'available; the recording supplies limited independent repeat days. Coverage rules are applied before estimating each clock-bin mean. '
    'Patterns show time-of-day association within this recording and should not be interpreted as evidence for an endogenous circadian '
    'rhythm without replication or a suitable experimental design.'),
'07_trip_investment':('Return frequency and time invested in trips',
    'The left panel compares each ant’s completed return rate and median trip duration, with Poisson and bootstrap intervals. Other panels '
    'show ant-level return rate and total minutes on completed trips per recording day, grouped by behavioral class. All classified ants '
    'are retained, including zero-trip ants. Grid-occupancy rules bridge position-state gaps up to 30 seconds and require anchored trips '
    'of at least 30 seconds. These 5,923 completed trips use broader gap rules than the 1,827 precise return-response events.'),
}


def description(name,index):
    if name=='07_return_response':
        if index%2==0:
            return ('Motion and contacts around nest return',
                'Curves align body motion, antennal motion and new-pair contact rate to the first observed outside-to-inside crossing. '
                'The title identifies the pooled cohort or behavioral class and gives event and ant counts. Events are averaged within '
                'ants before averaging ants; shaded intervals bootstrap ants. These precise returns use stricter position-gap criteria '
                'than the trip-investment analysis. Sleep labels do not select the return events. Contact-rate peaks describe temporal '
                'association with return, not evidence that return itself causes contact.')
        return ('Sleeping recipients contacted by returning ants',
            'Curves compare recipients contacted by a recent returner with matched no-contact controls and other contacts. Putative sleep '
            'uses fresh trailing-window motion labels whose body and antennal thresholds are calibrated from S0. Eligibility still requires '
            'ten continuously sleeping seconds before contact. Curves average events within ants, then ants; intervals require at least '
            'five contributing ants. Missing data reduce support with lag. Motion-defined awake fraction is partly definitional and has '
            'trailing-window delay; the body and antenna traces show the underlying response. This is an observational matched comparison.')
    note=NOTES[name]
    return note[index] if isinstance(note,list) else note


def label_figure(figure,name,index):
    title,caption=description(name,index)
    figure._report_title=title;figure._report_caption=caption
    # Sequential PDF/PNG figure numbers replace legacy section-number prefixes.
    for artist in [*(ax.title for ax in figure.axes),figure._suptitle]:
        if artist is not None:artist.set_text(re.sub(r'^\d+[a-z]?\.\s*','',artist.get_text()))


def add_pdf_description(figure):
    """Place a readable description below the full plot; keep vector artwork."""
    figure.canvas.draw();box=figure.get_tightbbox(figure.canvas.get_renderer())
    width,height=figure.get_size_inches();font=10
    # Wrap in physical units because figure widths range from 6 to 15 inches.
    line_width=max(50,int(box.width*72/(font*.52)))
    text=f'Figure {figure.number}. {figure._report_title}\n'+textwrap.fill(figure._report_caption,width=line_width)
    return figure.text(box.x0/width,(box.y0-.25)/height,text,ha='left',va='top',
        fontsize=font,linespacing=1.4,wrap=False)
