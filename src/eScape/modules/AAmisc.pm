package AAmisc;
use strict;
use lib '/Users/jgu/Research/Software/perl/modules';
use WinGrab;
use ArrayMisc;

sub getAAarray {
	my $AA = 'ACDEFGHIKLMNPQRSTVWY';
	my @AA = split('', $AA);
	return @AA;
}

sub getRESarray {
	my $AA = 'ACDEFGHIKLMNPQRSTVWY';
        my @AA = split('', $AA);
	my @RES;

	foreach (@AA) {
	   my $RES = AAtranslate($_);
	   push(@RES, $RES);
	}

        return @RES;
}

sub AAbinary {
	my ($aref) = @_;
	my @AA = @{$aref};
	my @AAbincode = '';

	# NEED TO FIGURE OUT DIRECTION OF ALIGNMENT INPUT IN ARRAY
	# return pos vs encoding. pos = x-axis, data = y-axis

	for my $i (0.. $#AA) {
	my $AA = $AA[$i];
	my @coder = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0); 

                      if ($AA eq 'A') { $coder[0] = 1; }
                      elsif ($AA eq 'C') { $coder[1] = 1; }
                      elsif ($AA eq 'D') { $coder[2] = 1; }
                      elsif ($AA eq 'E') { $coder[3] = 1; }
                      elsif ($AA eq 'F') { $coder[4] = 1; }
                      elsif ($AA eq 'G') { $coder[5] = 1; }
                      elsif ($AA eq 'H') { $coder[6] = 1; }
                      elsif ($AA eq 'I') { $coder[7] = 1; }
                      elsif ($AA eq 'K') { $coder[8] = 1; }
                      elsif ($AA eq 'L') { $coder[9] = 1; }
                      elsif ($AA eq 'M') { $coder[10] = 1; }
                      elsif ($AA eq 'N') { $coder[11] = 1; }
                      elsif ($AA eq 'P') { $coder[12] = 1; }
                      elsif ($AA eq 'Q') { $coder[13] = 1; }
                      elsif ($AA eq 'R') { $coder[14] = 1; }
                      elsif ($AA eq 'S') { $coder[15] = 1; }
                      elsif ($AA eq 'T') { $coder[16] = 1; }
                      elsif ($AA eq 'V') { $coder[17] = 1; }
                      elsif ($AA eq 'W') { $coder[18] = 1; }
                      elsif ($AA eq 'Y') { $coder[19] = 1; }
		      else { } # no change 		

		$AAbincode[$i] = [@coder];
	}
	return @AAbincode;
}

sub AAID_alpha {
	my $AA = $_[0];
	if ($AA =~ /\S{3}/) {$AA = AAtranslate($AA);}
	my $numID;

                      if ($AA eq 'A') { $numID = 1; }
                      elsif ($AA eq 'C') { $numID = 2; }
                      elsif ($AA eq 'D') { $numID = 3; }
                      elsif ($AA eq 'E') { $numID = 4; }
                      elsif ($AA eq 'F') { $numID = 5; }
                      elsif ($AA eq 'G') { $numID = 6; }
                      elsif ($AA eq 'H') { $numID = 7; }
                      elsif ($AA eq 'I') { $numID = 8; }
                      elsif ($AA eq 'K') { $numID = 9; }
                      elsif ($AA eq 'L') { $numID = 10; }
                      elsif ($AA eq 'M') { $numID = 11; }
                      elsif ($AA eq 'N') { $numID = 12; }
                      elsif ($AA eq 'P') { $numID = 13; }
                      elsif ($AA eq 'Q') { $numID = 14; }
                      elsif ($AA eq 'R') { $numID = 15; }
                      elsif ($AA eq 'S') { $numID = 16; }
                      elsif ($AA eq 'T') { $numID = 17; }
                      elsif ($AA eq 'V') { $numID = 18; }
                      elsif ($AA eq 'W') { $numID = 19; }
                      elsif ($AA eq 'Y') { $numID = 20; }
                      else { $numID = 21; } # no change

	return $numID;
}

sub AAID_Pscale {
	# "Engelman DM, Steitz TA, Goldman A. Ann Rev Biophys Biophysic Chem. 15(1986):330"
	# least polar, to most polar
        my $AA = $_[0];
        if ($AA =~ /\S{3}/) {$AA = AAtranslate($AA);}
        my $numID;

                      if ($AA eq 'F') { $numID = 1; }
                      elsif ($AA eq 'M') { $numID = 2; }
                      elsif ($AA eq 'I') { $numID = 3; }
                      elsif ($AA eq 'L') { $numID = 4; }
                      elsif ($AA eq 'V') { $numID = 5; }
                      elsif ($AA eq 'C') { $numID = 6; }
                      elsif ($AA eq 'W') { $numID = 7; }
                      elsif ($AA eq 'A') { $numID = 8; }
                      elsif ($AA eq 'T') { $numID = 9; }
                      elsif ($AA eq 'G') { $numID = 10; }
                      elsif ($AA eq 'S') { $numID = 11; }
                      elsif ($AA eq 'P') { $numID = 12; }
                      elsif ($AA eq 'Y') { $numID = 13; }
                      elsif ($AA eq 'H') { $numID = 14; }
                      elsif ($AA eq 'Q') { $numID = 15; }
                      elsif ($AA eq 'N') { $numID = 16; }
                      elsif ($AA eq 'E') { $numID = 17; }
                      elsif ($AA eq 'K') { $numID = 18; }
                      elsif ($AA eq 'D') { $numID = 19; }
                      elsif ($AA eq 'R') { $numID = 20; }
                      else { $numID = 21; } # no change

        return $numID;
}

sub AAID_Fscale {
        # Flexibilty index from Wiggle
	# "Gu J, Gribskov M, Bourne P. PLoS Computational Biology "
	# Most FFR to least FFR

        my $AA = $_[0];
        if ($AA =~ /\S{3}/) {$AA = AAtranslate($AA);}
        my $numID;

                      if ($AA eq 'E') { $numID = 1; }
                      elsif ($AA eq 'K') { $numID = 2; }
                      elsif ($AA eq 'Q') { $numID = 3; }
                      elsif ($AA eq 'R') { $numID = 4; }
                      elsif ($AA eq 'D') { $numID = 5; }
                      elsif ($AA eq 'P') { $numID = 6; }
                      elsif ($AA eq 'N') { $numID = 7; }
                      elsif ($AA eq 'S') { $numID = 8; }
                      elsif ($AA eq 'G') { $numID = 9; }
                      elsif ($AA eq 'A') { $numID = 10; }
                      elsif ($AA eq 'L') { $numID = 11; }
                      elsif ($AA eq 'T') { $numID = 12; }
                      elsif ($AA eq 'W') { $numID = 13; }
                      elsif ($AA eq 'H') { $numID = 14; }
                      elsif ($AA eq 'M') { $numID = 15; }
                      elsif ($AA eq 'Y') { $numID = 16; }
                      elsif ($AA eq 'F') { $numID = 17; }
                      elsif ($AA eq 'C') { $numID = 18; }
                      elsif ($AA eq 'V') { $numID = 19; }
                      elsif ($AA eq 'I') { $numID = 20; }
                      else { $numID = 21; } # no change

        return $numID;
}

sub AAID_Hscale {
       	# Hydrophobicity Scale 
	# "Palliser CC, Parry DA (2001) Proteins 42:243-255"
	# Most Hydrophobic to least hydrophobic

        my $AA = $_[0];
        if ($AA =~ /\S{3}/) {$AA = AAtranslate($AA);}
        my $numID;

                      if ($AA eq 'F') { $numID = 1; }
                      elsif ($AA eq 'I') { $numID = 2; }
                      elsif ($AA eq 'W') { $numID = 3; }
                      elsif ($AA eq 'L') { $numID = 4; }
                      elsif ($AA eq 'V') { $numID = 5; }
                      elsif ($AA eq 'M') { $numID = 6; }
                      elsif ($AA eq 'C') { $numID = 7; }
                      elsif ($AA eq 'Y') { $numID = 8; }
                      elsif ($AA eq 'A') { $numID = 9; }
                      elsif ($AA eq 'P') { $numID = 10; }
                      elsif ($AA eq 'H') { $numID = 11; }
                      elsif ($AA eq 'G') { $numID = 12; }
                      elsif ($AA eq 'T') { $numID = 13; }
                      elsif ($AA eq 'S') { $numID = 14; }
                      elsif ($AA eq 'Q') { $numID = 15; }
                      elsif ($AA eq 'N') { $numID = 16; }
                      elsif ($AA eq 'E') { $numID = 17; }
                      elsif ($AA eq 'D') { $numID = 18; }
                      elsif ($AA eq 'R') { $numID = 19; }
                      elsif ($AA eq 'K') { $numID = 20; }
                      else { $numID = 21; } # no change

        return $numID;
}


sub AAfreq {
	
	my ($aref, $href) = @_;
	my $seq;
	my @AA = @{ $aref };
	my %AA = %{ $aref };
	my %freq;
	die "Incomplete Subroutine";	
           foreach (@AA) {
                  print "$_ \n";
                  SWITCH: {
                                if (/A/) { ${$freq{"A"}}[$seq]++; next; };
                                if (/C/) { ${$freq{"C"}}[$seq]++; next; };
                                if (/D/) { ${$freq{"D"}}[$seq]++; next; };
                                if (/E/) { ${$freq{"E"}}[$seq]++; next; };
                                if (/F/) { ${$freq{"F"}}[$seq]++; next; };
                                if (/G/) { ${$freq{"G"}}[$seq]++; next; };
                                if (/H/) { ${$freq{"H"}}[$seq]++; next; };
                                if (/I/) { ${$freq{"I"}}[$seq]++; next; };
                                if (/K/) { ${$freq{"K"}}[$seq]++; next; };
                                if (/L/) { ${$freq{"L"}}[$seq]++; next; };
                                if (/M/) { ${$freq{"M"}}[$seq]++; next; };
                                if (/N/) { ${$freq{"N"}}[$seq]++; next; };
                                if (/P/) { ${$freq{"P"}}[$seq]++; next; };
                                if (/Q/) { ${$freq{"Q"}}[$seq]++; next; };
                                if (/R/) { ${$freq{"R"}}[$seq]++; next; };
                                if (/S/) { ${$freq{"S"}}[$seq]++; next; };
                                if (/T/) { ${$freq{"T"}}[$seq]++; next; };
                                if (/V/) { ${$freq{"V"}}[$seq]++; next; };
                                if (/W/) { ${$freq{"W"}}[$seq]++; next; };
                                if (/Y/) { ${$freq{"Y"}}[$seq]++; next; };
				
                  }
          }

}

sub AAfreqA {

        my ($aref) = @_;
        my @AA = @{ $aref };
        my @freq;
	
	for my $i (0 .. 19) {$freq[$i] = 0;}

        my $total = $#AA + 1;
	   # print "TOTAL : $total \n"; 
	   foreach (@AA) {
                  # print "$_ \n";
                  SWITCH: {
                                if (/A/) { $freq[0]++; next; };
                                if (/C/) { $freq[1]++; next; };
                                if (/D/) { $freq[2]++; next; };
                                if (/E/) { $freq[3]++; next; };
                                if (/F/) { $freq[4]++; next; };
                                if (/G/) { $freq[5]++; next; };
                                if (/H/) { $freq[6]++; next; };
                                if (/I/) { $freq[7]++; next; };
                                if (/K/) { $freq[8]++; next; };
                                if (/L/) { $freq[9]++; next; };
                                if (/M/) { $freq[10]++; next; };
                                if (/N/) { $freq[11]++; next; };
                                if (/P/) { $freq[12]++; next; };
                                if (/Q/) { $freq[13]++; next; };
                                if (/R/) { $freq[14]++; next; };
                                if (/S/) { $freq[15]++; next; };
                                if (/T/) { $freq[16]++; next; };
                                if (/V/) { $freq[17]++; next; };
                                if (/W/) { $freq[18]++; next; };
                                if (/Y/) { $freq[19]++; next; };

                  }
          }
	for my $i  ( 0 .. $#freq) {
		$freq[$i] = $freq[$i]/$total;
	}
	return @freq;
}


sub AAparse {
        my ($aref, $href) = @_;
        my @AoA = @{ $aref };
        my %AA = %{ $href };

           for my $i (0 .. $#AoA) {
			my $j = $#{$AoA[$i]};
			my $aa = $AoA[$i][0];
			my $data = $AoA[$i][$j];

                 	push( @{$AA{$aa}}, $data);
          }
		
	return %AA;
}

# usage:: $aa = AAmisc::AAtranslate($aa);
sub AAtranslate {
	my $aa = $_[0];
	SWITCH: {
        if ($aa=~/GLY/) { $aa='G'; return $aa; last SWITCH;}
        if ($aa=~/ALA/) { $aa='A'; return $aa; last SWITCH;}
        if ($aa=~/VAL/) { $aa='V'; return $aa; last SWITCH;}
        if ($aa=~/LEU/) { $aa='L'; return $aa; last SWITCH;}
        if ($aa=~/ILE/) { $aa='I'; return $aa; last SWITCH;}
        if ($aa=~/PRO/) { $aa='P'; return $aa; last SWITCH;}
        if ($aa=~/PHE/) { $aa='F'; return $aa; last SWITCH;}
        if ($aa=~/TYR/) { $aa='Y'; return $aa; last SWITCH;}
        if ($aa=~/TRP/) { $aa='W'; return $aa; last SWITCH;}
        if ($aa=~/CYS/) { $aa='C'; return $aa; last SWITCH;}
        if ($aa=~/MET/) { $aa='M'; return $aa; last SWITCH;}
        if ($aa=~/SER/) { $aa='S'; return $aa; last SWITCH;}
        if ($aa=~/THR/) { $aa='T'; return $aa; last SWITCH;}
        if ($aa=~/LYS/) { $aa='K'; return $aa; last SWITCH;}
        if ($aa=~/ARG/) { $aa='R'; return $aa; last SWITCH;}
        if ($aa=~/HIS/) { $aa='H'; return $aa; last SWITCH;}
        if ($aa=~/ASP/) { $aa='D'; return $aa; last SWITCH;}
        if ($aa=~/GLU/) { $aa='E'; return $aa; last SWITCH;}
        if ($aa=~/ASN/) { $aa='N'; return $aa; last SWITCH;}
        if ($aa=~/GLN/) { $aa='Q'; return $aa; last SWITCH;}
	if ($aa=~/[A-Z]{3}/) { $aa = 'X'; return $aa; last SWITCH;}
        if ($aa=~/G/) { $aa='GLY'; return $aa; last SWITCH;}
        if ($aa=~/A/) { $aa='ALA'; return $aa; last SWITCH;}
        if ($aa=~/V/) { $aa='VAL'; return $aa; last SWITCH;}
        if ($aa=~/L/) { $aa='LEU'; return $aa; last SWITCH;}
        if ($aa=~/I/) { $aa='ILE'; return $aa; last SWITCH;}
        if ($aa=~/P/) { $aa='PRO'; return $aa; last SWITCH;}
        if ($aa=~/F/) { $aa='PHE'; return $aa; last SWITCH;}
        if ($aa=~/Y/) { $aa='TYR'; return $aa; last SWITCH;}
        if ($aa=~/W/) { $aa='TRP'; return $aa; last SWITCH;}
        if ($aa=~/C/) { $aa='CYS'; return $aa; last SWITCH;}
        if ($aa=~/M/) { $aa='MET'; return $aa; last SWITCH;}
        if ($aa=~/S/) { $aa='SER'; return $aa; last SWITCH;}
        if ($aa=~/T/) { $aa='THR'; return $aa; last SWITCH;}
        if ($aa=~/K/) { $aa='LYS'; return $aa; last SWITCH;}
        if ($aa=~/R/) { $aa='ARG'; return $aa; last SWITCH;}
        if ($aa=~/H/) { $aa='HIS'; return $aa; last SWITCH;}
        if ($aa=~/D/) { $aa='ASP'; return $aa; last SWITCH;}
        if ($aa=~/E/) { $aa='GLU'; return $aa; last SWITCH;}
        if ($aa=~/N/) { $aa='ASN'; return $aa; last SWITCH;}
        if ($aa=~/Q/) { $aa='GLN'; return $aa; last SWITCH;}
	else {return $aa;}
        }
}

sub ClassificationCount {
        my ($aref, $href) = @_;
        my @AoA = @{ $aref };
        my %DATA = %{ $href };

           for my $i (0 .. $#AoA) {
                        my $j = $#{$AoA[$i]};
                        my $data = $AoA[$i][$j];

                        $DATA{$data}++;
            }
        return %DATA;
}

sub AAproperties {
	my ($res, $class) = @_;
        my $ans = 0; # 0 = false

	if ($class eq 'phobic1') {
	    if ($res =~ /[AVLIFYWTCMHKG]/) { 
		$ans = 1;
  	    }
	}
	elsif ($class eq 'phobic2') {	
	    if ($res =~ /[AVLIFYW]/) {
		$ans = 1;	
	    }
	}
	elsif ($class eq 'polar') {
	    if ($res =~	/[YWSTNQDEHKR]/) {
		$ans = 1;
	    }
	}
	elsif ($class eq 'small') {
	   if ($res =~	/[AVSTCNDGP]/) {
		$ans = 1;
	   }
	}
	elsif ($class eq 'tiny') {
	   if ($res =~ 	/[AGS]/) {
		$ans = 1;	
	   }
	}
	elsif ($class eq 'aliphatic') {
	   if ($res =~	/[ILV]/) {
		$ans = 1;
	   }
	}
	elsif ($class eq 'aromatic') {
	    if ($res =~ /[FYWH]/) {
		$ans = 1;
	    }
	}
	elsif ($class eq 'positive') {
	    if ($res =~ /[HKR]/) {
		$ans = 1;	
 	    }
	}
	elsif ($class eq 'negative') {
   	    if ($res =~ /[ED]/) {
	   	$ans = 1; 
	    }
	}
	elsif ($class eq 'charged') {
   	    if ($res =~ /[HKRED]/) {
		$ans = 1;		
	    }
	}
	elsif ($class eq 'conformational') {
	    if ($res =~ /[PG]/) {
		$ans = 1;
	    }
	}
	elsif ($class eq 'neutral') {
	    if ($res =~ /[STCMNQ]/) {
		$ans = 1;
	    }	
	}
	else {
		die "Incorrect type class see AAmisc module for options \n.";
	}	
	
	return $ans;
}



#usage @fasta = AAmisc::readFASTA($file);
sub readFASTA {
	my ($file) = @_;
	my @fasta;

	open (FASTA, $file) || die "Cannot open $file. \n" ;
	while (<FASTA>) {
		if (/^>/ || /^\s/) {next;}
		elsif (/^([A-Z]+)/) {
		    my $seq = $_;
		    chomp($seq);

                    my @tmp = split('',$seq);
		    while($tmp[$#tmp] =~ /\s/) {
			pop(@tmp);
	            }	
                    push(@fasta, @tmp);
		}
		else {
		    last;
		}
	}
	return @fasta;
}

sub readFASTA_ID {
        my ($file) = @_;
        my $id;

        open (FASTA, $file) || die "Cannot open $file. \n" ;
        while (<FASTA>) {
                if (/^>(.+)/) {
		   $id = $1; chomp($id);
		   last;
		}
	}
	return $id;
}

sub read_probcons {
        my ($file) = @_;
        my @fasta = ();
	my @MSA;
	my $id;

        open (MSA, $file) || die "Cannot open $file. \n" ;
        while (<MSA>) {
		
                if (/^>/ || /^\s/) {
		       if (/>(.+?)\s/) {
			   my $tmp = $1;

			   if ($#fasta > 0) {
                       	     push(@MSA,[($id, @fasta)]);
                             @fasta = ();
			   }
			   $id = $1;
			}
		}
                elsif (/^([-A-Z]+)/) {
                    my $seq = $_;
                    chomp($seq);

                    my @tmp = split('',$seq);
                    while($tmp[$#tmp] =~ /\s/) {
                        pop(@tmp);
                    }
                    push(@fasta, @tmp);
                }
                else {
		    last;
                }
        }
	# ADD LAST SEQUENCE TO @MSA
	push(@MSA, [($id, @fasta)]);

        return @MSA;
}

sub read_clustalW {
        my ($file) = @_;
        my @fasta = ();
        my %MSA;
	my @MSA;
        my $id;

        open (MSA, $file) || die "Cannot open $file. \n" ;
        while (<MSA>) {

                if (/^([.\w\d]+?)\s+([-A-Z]+?)\s/) {
                           my $id = $1;
			   my $seq = $2;
			   my @tmpseq = split('', $seq);
			
			   if (exists $MSA{$id}) {}
			   else {@{$MSA{$id}} = ()};

			   @{$MSA{$id}} = (@{$MSA{$id}}, @tmpseq);
                }
        }

	foreach (keys %MSA) {
		my $id = $_;
		my @fasta = @{$MSA{$id}};

		push (@MSA, [($id, @fasta)]);
	}

        return @MSA;
}

sub calcSEQID {
	my ($aref) = @_;
	my @MSA = @{$aref};
	my (@ID, @SEQ);
	my @SEQID;
	
	foreach (@MSA) {
	   my ($file, @tmp) = @{$_};
	   push(@ID, $file);
	   push(@SEQ, [@tmp]);
	}

	my @CALC;	
	my ($s1, $s2) = (0,1);
	my $nseq = 0;

	while ($nseq <= $#SEQ) {
		
		# COUNT ID BETWEEN SEQS
		my @SEQ1 = @{$SEQ[$s1]};
		my @SEQ2 = @{$SEQ[$s2]};

                my $IDcount = 0;
		for my $i (0 .. $#SEQ1) {
			if ($SEQ1[$i] eq $SEQ2[$i] && $SEQ1[$i] ne '-') {
				$IDcount++;
				# print "$SEQ1[$i] $SEQ2[$i] \n";
			}
		}
		
		# GET LENGTH OF SHORTEST OF TWO
		my $SEQ1 = join('', @SEQ1); $SEQ1 =~ s#\-##g;
		my $SEQ2 = join('', @SEQ2); $SEQ2 =~ s#\-##g;

		# print "SEQ1: $SEQ1 \nSEQ2: $SEQ2 \n";

		@SEQ1 = split('', $SEQ1);
		@SEQ2 = split('', $SEQ2);

		my $nAA;
		if ($#SEQ1 > $#SEQ2) { 
			$nAA = $#SEQ2 + 1;
		}
		else { $nAA = $#SEQ1 + 1; }

		my $seqID = $IDcount/$nAA;
		print "IDcount: $IDcount nAA: $nAA \n";
	
		push(@SEQID, [($ID[$s1], $ID[$s2], $seqID)]);

		$nseq++;
		$s1++,$s2++;  if ($s2 > $#SEQ) {$s2 = 0;}
	}
	return @SEQID;

}

sub alignFASTA {
	my ($aref1, $aref2) = @_;
	my @FASTA1 = @{$aref1};
	my @FASTA2 = @{$aref2};
	my @WIN1 = winFASTA(3, \@FASTA1);
	my @WIN2 = winFASTA(3, \@FASTA2);
	my @INDEX;
	my ($index1, $index2) = (0,0);

	# print "FASTA1: $#FASTA1 $FASTA1[0]  FASTA2: $#FASTA2 $FASTA2[0]\n";
	# print "WIN1 : $#WIN1  $WIN1[0][0] WIN2: $#WIN2 $WIN2[0][0] \n";

	# SHIFT ZERO ENDS - POTENTIAL BUG ZONE FOR WINDOW > 3
	shift(@WIN1); shift(@WIN2);
	my @end1 = (@{$WIN1[$#WIN1]}); shift(@end1); push(@end1,1);
	my @end2 = (@{$WIN2[$#WIN2]}); shift(@end2); push(@end2,1);
	push(@WIN1, [@end1]);	
        push(@WIN2, [@end2]);

	my $go1 = 1; # go flag = 0 - GO; 1 - STOP
	my $go2 = 1;

	# COLLECTING INDEX FOR COMPARISON OF 2 ARRAYS WITH SEQUENCES
	while ($#WIN1 > -1 || $#WIN2 > -1) {

	   my ($win1, $win2, $AA1, $AA2);

	   # READ FIRST WINDOW FROM FASTA1
	   if ($#WIN1 > -1)  {
                $AA1 = $WIN1[0][$#{$WIN1[0]}];
                $win1 = join('', @{$WIN1[0]});
                # $win1 =~ s#0##;
                $go1 = 0;
		# print "FINISH READING FIRST WINDOW. \n";
	   }

	   # READ FIRST WINDOW FROM FASTA2
	   if ($#WIN2 > -1) { 
		$AA2 = $WIN2[0][$#{$WIN2[0]}];
		$win2 = join('', @{$WIN2[0]});
		# $win2 =~ s#0##;
	   }
	   # print "AA1: $AA1 AA2: $AA2 WIN1: $win1 n: $#WIN1 WIN2: $win2 n: $#WIN2\n";

	   # COMPARING SEQUENCES - CASE SCENARIOS:	
	   # EXACT MATCH
	   #my $AA11 = $WIN1[0][0];
	   #my $AA12 = $WIN2[0][0];
		
	   #if ($win1 =~ /$win2/ || $AA11 eq $AA12) {
	   if ($win1 eq $win2) {
		#  print "PUSH4 INDEX1: $index1 INDEX2: $index2\n";
		push(@INDEX, [($index1, $index2)]);
		$index1++;
		$index2++;
		shift(@WIN1);
		shift(@WIN2);
		$go2 = 0;
	   }
	   # REMAINING C TERMINAL END FOR FASTA1
	   elsif ($#WIN2 <  0 && $#WIN1 != -1) {
		# print " PUSH 0 INDEX1 : $index1 \n";
		push(@INDEX, [($index1, 'GAP')]);
		$index1++;
		shift(@WIN1);
	   }
           # REMAINING C TERMINAL END FOR FASTA2
           elsif ($#WIN1 < 0) {
		# print "PUSH 5 NO MORE INDEX1: $index1 INDEX2: $index2 WIN1: $#WIN1\n";
                push(@INDEX, [('GAP', $index2)]);
                $index2++;
                shift(@WIN2);
           }
           # LEADING N TERMINAL END FOR FASTA1
           elsif ($go1 == 0 && $go2 == 1)  {
		# print "PUSH2 INDEX1 : $index1 \n";
                push(@INDEX, [($index1, 'GAP')]);
                $index1++;
		shift(@WIN1);
           }
	   # MISSING ATOMS IN FASTA1 
	   else {
		$go1 = 1;
	
		# SHIFT ARRAY UNTIL UNMATCHED REGIONS
		# COMPARE AT N TERMINAL REGIONS (FIRST RESIDUE)
		# POSSIBLY REDUNDANT NOW THAT NEW IF CRITERIA ADDED IN MAIN IF
		$AA1 = '';
		$AA2 = '';

		while ($AA1 eq $AA2) {

			if ($#WIN1 < 0) {last;}
			if ($#WIN2 < 0) {last;}
	
                        $AA1 = $WIN1[0][0];
                        $AA2 = $WIN2[0][0];
			# print "PUSH3 INDEX1 : $index1 WIN1: $#WIN1\n";
			push(@INDEX, [($index1, $index2)]);
			$index1++; $index2++;
			shift(@WIN1);
			shift(@WIN2);
			
		}

                # SEARCH FOR NEXT MATCH IN FASTA
		# CHECK IF AT END OF SEQUENCE	
		if ($#WIN1 < 0 ) { next; }
		   
		my $WIN1 = join('', @{$WIN1[0]});
	        my $WIN2 = join('', @{$WIN2[0]});

		while ($AA1 ne $AA2 && $#WIN2 > -1) {
			$AA1 = $WIN1[0][0];
			$AA2 = $WIN2[0][0];	
			push(@INDEX, [('GAP', $index2)]);
			$index2++;
	
			shift(@WIN2);
			$WIN2 = join('', @{$WIN2[0]});
			if ($WIN1 eq $WIN2) {last;}
		}	
	   }
	
	} # END SEQ ALIGN

	return @INDEX;	
}

sub SegmentLength {
	# Returns a Hash with number of segment length observed.
	my ($aref, $href) = @_;
	my @AoA = @{ $aref };
	my %DATA = %{ $href };

	my $l = 0;
	my $id = '';
        for my $i (0 .. $#AoA) {
                my $j = $#{$AoA[$i]};
                my $data = $AoA[$i][$j];

		if ($l == 0 || $id == $data) {
			$l++;
			$id = $data;
		}
		else {
			$DATA{$id}{$l}++;
			$l = 0;
			$id = $data;
		}	
       }
        return %DATA;
}

sub winFASTA {
        # Grabbing sequence within window size.
        my ($win, $aref) = @_;
        my @a = @{ $aref } ;    # ENTIRE LENGTH OF SEQUENCE

        my @Aseg;

                ## CREATING SVM INPUT DATA FOR GIVEN POS AND WINDOW SIZE

                my @win = WinGrab::get_win($win);

                for my $i (0 .. $#a) {
                        my @seg = ();
                        for my $j (0 .. $#win) {
                                my $pos = $i + $win[$j];

                                if ($pos < 0 || $pos> $#a) {
                                        @seg = (@seg, 0);
                                }
                                else {
                                        @seg = (@seg, $a[$pos]);
                                }
                        }
                        $Aseg[$i] = [@seg];
                }
                return @Aseg;
}


1;
