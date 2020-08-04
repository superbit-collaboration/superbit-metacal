/* annular.c: make a radial mass/shear profile
   Created Wed Mar 18 1998 DMW

   
   This is a rewrite of Tony's rdens [now dropped; see last entry below],
   with these advantages:
   1. much faster: reads the catalog only once and sorts in-memory
   2. flexible command line
   3. reads fiat catalogs
   
   Later I added Ian's binmom algorithm, which has the advantage
   of uncorrelated errors, but doesn't use all the information available.
   They are output side by side and you can take your pick.

   980817: added log-spaced binning option
   980820: corrected rdens errorbars to include rmax/r factor

   9902: added, but not fully tested, Gary's seeing correction and
   optimal weights.  By default no correction is made, and in any case
   the rdens algorithm is unchanged.

   990420: added -d option
   990514: added -o option

   990823: seeing correction is now applied to rdens algorithm
   also. Seeing correction is now a polynomial fit to (x,y) rather
   than constant.  Put SCCS version in help string.  

   990927: seeing correction now goes through new fiat_zlookup and
   fiat_read_zfit funcs, rather than custom funcs.  Major bug fixes:
   add to rdens sums AFTER checking for bizarre seeing corrections, and 
   apply ellipticity criteria AFTER seeing correction.  Minor fixes:
   more throrough checking for bizarre objects: negative sigma_e, negative 
   size, <rho^4> >4 or <0, etc.  Updated SIGMA_SHAPE_SQ.

   991027: major update.  Can now read e components rather than
   moments from cat--interpretation depends on number of columns
   given.  Can also go with or without weighting/seeing correction,
   depending on same.  Tested and gives results identical to previous
   within roundoff error. Include more info in output header and use
   SCCS keywords to identify version.

   010222: added -D, which is a bit of a kludge.

   050104: major change to output shear instead of ellipticity.  This
   involves one factor of 2 and one "responsivity" factor of 1-<e^2>
   (one-component ellipticity).  <e2> can only be computed after
   examining all the data, so I no longer go thru in just one pass,
   and can now output smaller radii first (major impact on flow
   control).  Add -R option to manually input responsivity.  Remove
   seeing-correction code (-p -and -d options) because that function
   is now served better by seeingcorrect.c.  Also drop the rdens algorithm
   because I'm not sure how to apply the responsivity correction to it.

   After this, will check into CVS, so look for comments in CVS log
   rather than here.

 */

static char *cvsversion = "$Revision: 1.1 $";
static char *usage = "Usage: annular [-Dv] [-c columns] [-f filter] [-s startrad] [-e endrad]\n\
       [-n nbins] [-m minell] [-x maxell] [-R <e_i^2>] fiatfile xcenter ycenter\n\
\n\
  Prints shear vs. radius to stdout.  Options:\n\
   -c: read in named columns (default \"x y ixx iyy ixy\"). Can also use\n\
       e components (e.g. \"x y e1 e2\").  \n\
   -D: dump a list of object positions, e_tan, and e_x to stdout and exit.\n\
       Useful for checking geometry.\n\
   -f: filter catalog first (same usage as fiatfilter)\n\
   -R: manually specify <e_i^2> for responsivity\n\
   -v: verbose\n\
\n\
   nbins<0 will produce abs(nbins) logarithmically-spaced bins\n\
   all distances are in pixels\n\
   defaults: -s 0 -e 3000 -n 15 -m 0.0 -x 1.0\n";
static char *warning = "# WARNING: This version of annular is significantly changed\n\
# from all previous versions (which do not have a CVS identifier).\n\
# Output is now shear, not ellipticity!!!  This includes a factor of 2\n\
# ellipticity -> shear conversion and a 10%ish responsivity correction\n\
# which is computed from the data: 1-<e_i^2>.  Use the -R option if you\n\
# don't like the on-the-fly computation.  The rdens output has been dropped\n\
# until I figure out how to responsivity-correct it.\n";



#include <math.h>
#include <time.h>
#include <fiat.h>
#include <stdlib.h>
#include <unistd.h>		/* for unlink() */
#include <strings.h>
#include <string.h>


typedef struct {
  /* for binmom, with Gary's weighting scheme */
  float e1,e2;  /* tangential and 45-degree ellipticities */
  float e1sq,e2sq;  /* same, squared, for accumulating the variance */
  float e1_wtd,e2_wtd;  /* same but weighted with Gary's scheme */
  float e;      /* total ellipticity */
  float wt,r1,r2,var1,var2;  
  float rad; /* mean radius of objects in this bin */
  long n;    /* number in bin */
} bin_t;


/* 
 * funcs
 */
int column_count(char *str);


int main(int argc, char **argv)
{
  long i,n,ntot,ncols;
  int val;
  float f,m,r,r2,dx,dy,ixx,iyy,ixy,x,y,sigma_e,sherr1,sherr2;
  float wt,sin2phi,cos2phi;
  float e;          /* scalar ellipticity */
  float e1,e2,ex,ey;		 /* ellipticity components: x,y refer to
				  image, while 1,2 are about cluster center*/
  float responsivity;
  FILE *in;
  char tmpname[L_tmpnam];
  time_t clock;

  /* binning vars with reasonable BTC defaults*/
  bin_t *bins;
  int nbin=15,indx;
  float rmin=0.0,rmax=3000.0,binsize;

  /* getopt vars */
  char ch,Dflag=0,verbose,errflag,*colstr=NULL;
  float xc,yc,minell,maxell;  

  /* filter vars */
  char *filter, longstr[1024], *catname;

  /* responsivity computation */
  float e_ms=-1.0; /* initial flag so we know if user changed it */
  float e_sum=0.0;
  long n_esum=0;

  /* set up defaults */
  filter = NULL;
  minell=0.0;
  maxell=1.0;
  verbose=errflag=0;

  /* parse command line */
  while ((ch = getopt(argc,argv,"Dvc:e:f:m:n:s:x:R:")) != EOF)
    switch (ch) {
    case 'D':
      Dflag=1;  break;
    case 'v':
      verbose=1;  break;
    case 'c':
      colstr=strdup(optarg);
      break;
    case 'e':
      if(sscanf(optarg,"%f",&rmax)!=1)
	errflag=1;
      break;
    case 'f':
      filter=strdup(optarg);
      break;
    case 'm':
      if(sscanf(optarg,"%f",&minell)!=1)
	errflag=1;
      break;
    case 'n':
      if(sscanf(optarg,"%d",&nbin)!=1)
	errflag=1;
      break;
    case 'R':
      if(sscanf(optarg,"%f",&e_ms)!=1)
	errflag=1;
      if(e_ms<0){
	fprintf(stderr,"User error: e_ms must be >0!\n");
	errflag=1;
      }
      break;
    case 's':
      if(sscanf(optarg,"%f",&rmin)!=1)
	errflag=1;
      break;
    case 'x':
      if(sscanf(optarg,"%f",&maxell)!=1)
	errflag=1;
      break;
    case '?':
      errflag=1; break;
    }
  if( (argc-optind!=3) || errflag ||
      (sscanf(argv[optind+1],"%f",&xc)!=1) ||
      (sscanf(argv[optind+2],"%f",&yc)!=1)) {
    fprintf(stderr,"%s",usage);
    fprintf(stderr,"CVS: %s",cvsversion);
    fprintf(stderr,"%s",warning);
    exit(1);
  }
  if(rmin<0.0){
    fprintf(stderr,"Sorry, startrad<0 is nonsensical!\n");
    exit(2);
  }
  if(nbin<0 && rmin==0.0){
    fprintf(stderr,"Sorry, can't do log intervals with startrad==0!\n");
    exit(2);
  }

  /*
   * Set up catalog columns to read. if colstr is something like "x y
   *  e1 e2", plan on reading ellipticity components; otherwise use
   *  moments.  So count number of cols in str. 
   */
  if(colstr==NULL)
    colstr = strdup("x y ixx iyy ixy");
  ncols = column_count(colstr);
  if(ncols==4){
    if(verbose)
      fprintf(stderr,"Using moments.\n");
  }
  else if(ncols==5){
    if(verbose)
      fprintf(stderr,"Using ellipticity components.\n");
  }
  else{
    fprintf(stderr,"annular: must use only 4 or 5 input columns, exiting.\n");
    exit(3);
  }

  /* set up binning parameters */
  binsize = (rmax-rmin)/nbin;


  /* now we know how many bins; alloc and initialize memory */
  if((bins=malloc(abs(nbin)*sizeof(bin_t)))==NULL){
   fprintf(stderr,"Malloc failed, exiting\n"); 
   exit(2); 
  }
  bzero((void *)bins,abs(nbin)*sizeof(bin_t));


  /* Filter the catalog first if necessary. */
  if(filter!=NULL){
    tmpnam(tmpname);
    if(verbose)
      sprintf(longstr,"fiatfilter -v '%s' %s >%s",filter,argv[optind],tmpname);
    else
      sprintf(longstr,"fiatfilter '%s' %s >%s",filter,argv[optind],tmpname);
    if(system(longstr)!=0){
      fprintf(stderr,"fiatmap: %s failed, exiting",longstr);
      exit(1);
    }
    catname=strdup(tmpname);
  }
  else
    catname=strdup(argv[optind]);
  



  /* read in catalog header */
  if((in=fopen(catname,"r"))==NULL){
    fprintf(stderr,"Can't open file %s, exiting.\n",catname);
    if(filter!=NULL){
      sprintf(longstr,"/bin/rm -f %s",catname);
      system(longstr);
    }
    exit (2);
  }
  fiat_read_header(in);

  if(Dflag){
    /* print header for dumped catalog */
    printf("# fiat 1.0\n# creator =  ");
    for(i=0;i<argc;i++)
      printf("%s ",argv[i]);

    clock=time((time_t *)0);
    printf("# date = %s",ctime(&clock)); /* ctime appends \n automatically */   
    printf("# ttype1 = x\n");
    printf("# ttype2 = y\n");
    printf("# ttype3 = r / distance from center, pixels\n");
    printf("# ttype4 = e1 / tangent to cluster\n");
    printf("# ttype5 = e2 / 45 degrees to cluster\n");
  }


  /* 
   * read in catalog and compute tangential ellipticities as we go
   */
  if(verbose)
    fprintf(stderr,"Reading in catalog...");
  for (i=0; i<fiat_nentries; i++) {

    if(ncols==4)
      val=fiat_scanf(in,colstr,"%f %f %f %f",&x,&y,&ex,&ey);
    else if(ncols==5)
      val=fiat_scanf(in,colstr,"%f %f %f %f %f",&x,&y,&ixx,&iyy,&ixy);
    else{
      fprintf(stderr,"annular: impossible number of columns, exiting\n");
      exit(1);
    }
    if(val!=ncols){
      fprintf (stderr, "annular: %s: failure in fiat_scanf, object number %ld, return value %d\n",
               catname,i,val);
      if(filter!=NULL){
	unlink(catname);
      }
      exit(3);
    }


    /* 
     * find radius bin; skip this object if not within radius limits
     */
    dx = x-xc;
    dy = y-yc;
    r2 = dx*dx + dy*dy;
    r = sqrt(r2);
    if( (r>rmax) || (r<rmin) )
      continue;
    if(nbin>0)
      indx = (r-rmin)/binsize;
    else{
      /* log-spaced bins */
      indx = log(r/rmin)/log(rmax/rmin)*abs(nbin);
      /*fprintf(stderr,"r: %.2f indx: %d\n",r,indx);*/
    }

       
    /* 
     * if necessary, check validity of moments and convert 
     * to ellipticity components 
     */
    if(ncols==5){
      m = ixx+iyy;
      if(!(m>0.0)){ /* the weird syntax also catches NaNs */
	fprintf(stderr,"discarding object at %.2f,%.2f because ixx+iyy=%g\n",x,y,m);
	continue;
      }
      else{
	ex = (ixx-iyy)/m;
	ey = 2.0*ixy/m;
      }
    }
    e = hypot(ex,ey);

    /* 
     * apply ellipticity criteria. Weird syntax is to catch NaNs.
     *
     */
    if( !(e>=minell) || !(e<=maxell) )
      continue;

    /* 
     * rotate ellipticity components to be tangential about cluster.
     * Note that e2 is -1 * the usual rotation result, for backward
     * compatibility.  The sign of e2 doesn't matter physically, as it
     * is a null test.
     */
    cos2phi = (dy*dy - dx*dx) / r2;
    sin2phi = -2.0*dx*dy / r2;
    e1 = ex*cos2phi + ey*sin2phi;
    e2 = ex*sin2phi - ey*cos2phi;
    

    /* now would be a good time to apply weights--should make provision
       for inputting a weighted catalog */

    if(Dflag){
      /* dump rotated ellipticities */
      printf("%9.2f %9.2f %9.2f %.2f %.2f\n",x,y,r,e1,e2);
    }

    /* 
     * valid object: accumulate sums
     */
    bins[indx].e1 += e1;
    bins[indx].e2 += e2;
    bins[indx].e1sq += e1*e1;
    bins[indx].e2sq += e2*e2;
    
    if(e_ms<0){
      /* we need to compute the overall mean-square ellipticity for
	 the responsivity correction.  It's per-component, but let's
	 use both components for better S/N, and then later divide by 
	 sqrt(2). Just for kicks, let's also keep track of the mean */
      e_sum += e1*e1 + e2*e2;
      n_esum++;
    }

    /* other sums */
    bins[indx].n++;
    bins[indx].rad += r;

  } /* loop over objects in cat */

  if(e_ms<0){
    /* Finish computing the overall mean-square ellipticity and
       the responsivity correction--see above. */
    e_ms = e_sum/n_esum/sqrt(2.0);
  }
  responsivity = 1.0/(1.0-e_ms);

  if(verbose){
    fprintf(stderr,"read %ld objects\n",i);
    fprintf(stderr,"e_ms = %.3f, responsivity = %.3f\n",e_ms,responsivity);
  }

  if(Dflag)
    exit(0);

    /* output header */
  printf("# fiat 1.0\n# creator = ");
  for(i=0;i<argc;i++)
    printf("%s ",argv[i]);
  printf("\n# CVS: %s\n",cvsversion);
  printf("%s",warning);
  printf("# e_ms = %.3f / ",e_ms);
  if(n_esum>0){
    printf("computed on the fly\n");
  }
  else{
    printf("supplied by user\n");
  }
  clock=time((time_t *)0);
  printf("# date = %s",ctime(&clock)); /* ctime appends \n automatically */   
  printf("# ttype1 = radius / pixels\n");
  printf("# ttype2 = n / number of objects for shear calculation\n");
  printf("# ttype3 = shear1 / tangential \n");
  printf("# ttype4 = shear2 / 45-degree \n");
  printf("# ttype5 = err_shear1\n");
  printf("# ttype6 = err_shear2\n");


  /* step thru bins doing final arithmetic and output */
  for(i=0;i<abs(nbin);i++){
    float r;

    /* use mean radius of objects actually in this bin.
       This avoids overestimating the radius of any bins
       that go off the edge of the image/catalog. */
    r = bins[i].rad / bins[i].n;

    /* calculate unweighted shear: apply factor of 2 to convert
       ellipticity to shear, and apply responsivity correction.*/
    n = bins[i].n;

    /*
    STOPPED HERE. PROPOAGATE these corrections thru the below eqns!!!
    */
    sherr1 = sqrt((bins[i].e1sq - bins[i].e1*bins[i].e1/n)/(n-1.0)/(n-1.0));
    sherr2 = sqrt((bins[i].e2sq - bins[i].e2*bins[i].e2/n)/(n-1.0)/(n-1.0));
    bins[i].e1 /= n;
    bins[i].e2 /= n;
    bins[i].e /= n;

    printf("%.3g %7ld %9g %9g %9g %9g",
	   r,bins[i].n,bins[i].e1,bins[i].e2, /* annular stuff */
	   sherr1,sherr2, /* err in unwtd shear*/
	   bins[i].e );		/* mean ellipticity */
    

    if( (ncols==6) || (ncols==7) ){
      /* weighted shear and variance, and responsivity */
      bins[i].r1 /= bins[i].n;
      bins[i].r2 /= bins[i].wt;
      bins[i].e1_wtd /= bins[i].wt/bins[i].r1;
      bins[i].e2_wtd /= bins[i].wt/bins[i].r2;
      bins[i].var1 = sqrt(bins[i].var1)/bins[i].wt/bins[i].r1;
      bins[i].var2 = sqrt(bins[i].var2)/bins[i].wt/bins[i].r2;

      printf(" %9g %9g %9g %9g %9g %9g\n",	  
	     bins[i].e1_wtd,bins[i].var1,
	     bins[i].e2_wtd,bins[i].var2,
	     bins[i].r1,bins[i].r2);
    }
    else{
      printf("\n");
    }

  } /* loop over bins */
  

  /* cleanup */
  if(filter!=NULL){
    sprintf(longstr,"/bin/rm -f %s",catname);
    system(longstr);
  }
  free(bins);
  return 0;
}



int column_count(char *str)
{
  char *ptr;
  int ncols = 1;

  ptr = str;
  while( (ptr = strchr(ptr,' '))!=NULL){
    /* advance to next column name */
    while(ptr[0] == ' ')
      ptr++;
    ncols++;
    /*fprintf(stderr,"%d cols, at %s\n",ncols,ptr);*/
  }

  return ncols;
}
