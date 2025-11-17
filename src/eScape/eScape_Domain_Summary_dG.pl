#!/usr/bin/perl
#
# Grab dGN summary for eScape_Domain
# Window size is hardcoded.
#
#
# JDB 6/12/25
# Added FindBin to automatically locate modules/libraries
#
use strict;
use FindBin;
use lib "$FindBin::Bin/modules";
use ArrayMisc;

my @WIN = (1, 5, 11, 15, 21, 31, 51, 81);
my %DATA;

# GRAB DATA FROM FILES FOR SUMMARY
while (my $avgfile = <*avgpred>) {

	# ALSO GRAB sdpred and GET WINSIZE
	my ($sdfile, $id, $winsize) = INITIALIZE($avgfile);
	
	# READ DATA FROM avgpred AND sdpred	
	my @AVG = ArrayMisc::readAoA($avgfile);
	my @SD = ArrayMisc::readAoA($sdfile);

	@{$DATA{$id}{'avg'}{$winsize}} = ArrayMisc::getCOL(\@AVG, 0);
        @{$DATA{$id}{'sd'}{$winsize}} = ArrayMisc::getCOL(\@SD, 0);
	
}

# PRINT SUMMARY
foreach my $id (keys %DATA) {
foreach my $T ('avg', 'sd') {
	my @OUT;
	foreach my $W (@WIN) {
		my $LABEL = "win$W";
		
		print "$id $T $W \n";
		my @tmp = ($LABEL, @{$DATA{$id}{$T}{$W}});
		push(@OUT, [@tmp]);
	}
	@OUT = ArrayMisc::transposeAoA(\@OUT);

	my $outfile = "$id\_eScapeDomain.dGN_$T\_summary";
	ArrayMisc::printAoA2file($outfile, \@OUT);
	
}
}

#====================== END OF MAIN SCRIPT

#====================== START SUBROUTINES

sub INITIALIZE {
	my ($file1) = @_;
	my ($file2, $id, $winsize);

        my $file2 = $file1;
        $file2 =~ s#avgpred$#sdpred#;
        if ($file1 =~ /(.+?)\..+domain(\d+)_/) {
		$id = $1;
		$winsize = $2;
	}
	my @a = ($file2, $id, $winsize);	
	return @a;
}
