function [err, timepoints, species_out, observables_out] = model_even_smaller( timepoints, species_init, parameters, suppress_plot )
%MODEL_EVEN_SMALLER Integrate reaction network and plot observables.
%   Integrates the reaction network corresponding to the BioNetGen model
%   'model_even_smaller' and then (optionally) plots the observable trajectories,
%   or species trajectories if no observables are defined. Trajectories are
%   generated using either default or user-defined parameters and initial
%   species values. Integration is performed by the MATLAB stiff solver
%   'ode15s'. MODEL_EVEN_SMALLER returns an error value, a vector of timepoints,
%   species trajectories, and observable trajectories.
%   
%   [err, timepoints, species_out, observables_out]
%        = model_even_smaller( timepoints, species_init, parameters, suppress_plot )
%
%   INPUTS:
%   -------
%   species_init    : row vector of 32 initial species populations.
%   timepoints      : column vector of time points returned by integrator.
%   parameters      : row vector of 37 model parameters.
%   suppress_plot   : 0 if a plot is desired (default), 1 if plot is suppressed.
%
%   Note: to specify default value for an input argument, pass the empty array.
%
%   OUTPUTS:
%   --------
%   err             : 0 if the integrator exits without error, non-zero otherwise.
%   timepoints      : a row vector of timepoints returned by the integrator.
%   species_out     : array of species population trajectories
%                        (columns correspond to species, rows correspond to time).
%   observables_out : array of observable trajectories
%                        (columns correspond to observables, rows correspond to time).
%
%   QUESTIONS about the BNG Mfile generator?  Email justinshogg@gmail.com



%% Process input arguments

% define any missing arguments
if ( nargin < 1 )
    timepoints = [];
end

if ( nargin < 2 )
    species_init = [];
end

if ( nargin < 3 )
    parameters = [];
end

if ( nargin < 4 )
    suppress_plot = 0;
end


% initialize outputs (to avoid error msgs if script terminates early
err = 0;
species_out     = [];
observables_out = [];


% setup default parameters, if necessary
if ( isempty(parameters) )
   parameters = [ 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.3, 0.05, 3.0, 0.01, 0.05, 0.2, 0.8, 0.02, 0.015, 0.035, 0.05, 0.12, 0.15, 0.5, 0.06, 0.05, 0.08, 0.08, 0.2, 0.3, 0.6, 4.0, 1.0, 0.5, 0.1, 250.0, 40.0, 45.0, 30.0, 250.0, 50.0 ];
end
% check that parameters has proper dimensions
if (  size(parameters,1) ~= 1  ||  size(parameters,2) ~= 37  )
    fprintf( 1, 'Error: size of parameter argument is invalid! Correct size = [1 37].\n' );
    err = 1;
    return;
end

% setup default initial values, if necessary
if ( isempty(species_init) )
   species_init = initialize_species( parameters );
end
% check that species_init has proper dimensions
if (  size(species_init,1) ~= 1  ||  size(species_init,2) ~= 32  )
    fprintf( 1, 'Error: size of species_init argument is invalid! Correct size = [1 32].\n' );
    err = 1;
    return;
end

% setup default timepoints, if necessary
if ( isempty(timepoints) )
   timepoints = linspace(0,10,20+1)';
end
% check that timepoints has proper dimensions
if (  size(timepoints,1) < 2  ||  size(timepoints,2) ~= 1  )
    fprintf( 1, 'Error: size of timepoints argument is invalid! Correct size = [t 1], t>1.\n' );
    err = 1;
    return;
end

% setup default suppress_plot, if necessary
if ( isempty(suppress_plot) )
   suppress_plot = 0;
end
% check that suppress_plot has proper dimensions
if ( size(suppress_plot,1) ~= 1  ||  size(suppress_plot,2) ~= 1 )
    fprintf( 1, 'Error: suppress_plots argument should be a scalar!\n' );
    err = 1;
    return;
end

% define parameter labels (this is for the user's reference!)
param_labels = { 'TGFb_0', 'IL6_0', 's_TGFb', 'b_TGFb', 's_IL6', 'b_IL6', 'kf_il6_bind', 'kr_il6_bind', 'k_act_il6r', 'k_deact_il6r', 'kf_tgfb_bind', 'kr_tgfb_bind', 'k_act_tgfbr', 'k_deact_tgfbr', 'k_phos_smad3', 'k_dephos_smad3', 'kr_smad3_bind', 'kf_stat3_bind', 'k_phos_stat3', 'kf_dimer', 'k_dephos_stat3d', 'kr_dimer', 'k_dephos_stat3p', 'kf_smad3_bind', 'kr_s3stat3d', 'k_dephos_s3stat3d', 'kf_s3s4', 'kf_s3stat3d', 'kf_pka_bind', 'k_phos_pka', 'k_off_pka', 'IL6R_0', 'TGFbR_0', 'SMAD3_0', 'SMAD4_0', 'STAT3m_0', 'PKA_0' };



%% Integrate Network Model
 
% calculate expressions
[expressions] = calc_expressions( parameters );

% set ODE integrator options
opts = odeset( 'RelTol',   1e-8,   ...
               'AbsTol',   0.0001,   ...
               'Stats',    'off',  ...
               'BDF',      'off',    ...
               'MaxOrder', 5   );


% define derivative function
rhs_fcn = @(t,y)( calc_species_deriv( t, y, expressions ) );

% simulate model system (stiff integrator)
try 
    [~, species_out] = ode15s( rhs_fcn, timepoints, species_init', opts );
    if(length(timepoints) ~= size(species_out,1))
        exception = MException('ODE15sError:MissingOutput','Not all timepoints output\n');
        throw(exception);
    end
catch
    err = 1;
    fprintf( 1, 'Error: some problem encountered while integrating ODE network!\n' );
    return;
end

% calculate observables
observables_out = zeros( length(timepoints), 12 );
for t = 1 : length(timepoints)
    observables_out(t,:) = calc_observables( species_out(t,:), expressions );
end


%% Plot Output, if desired

if ( ~suppress_plot )
    
    % define plot labels
    observable_labels = { 'Free_IL6_obs', 'Free_TGFb_obs', 'IL6R_Active', 'TGFbR_Active', 'pSMAD3_obs', 'pSTAT3_obs', 'STAT3d_active_obs', 'S3S4_complex_obs', 'S3STAT3d_complex_obs', 'PKA_active', 'Total_SMAD3_obs', 'Total_STAT3_obs' };

    % construct figure
    plot(timepoints,observables_out);
    title('model_even_smaller observables','fontSize',14,'Interpreter','none');
    axis([0 timepoints(end) 0 inf]);
    legend(observable_labels,'fontSize',10,'Interpreter','none');
    xlabel('time','fontSize',12,'Interpreter','none');
    ylabel('number or concentration','fontSize',12,'Interpreter','none');

end


%~~~~~~~~~~~~~~~~~~~~~%
% END of main script! %
%~~~~~~~~~~~~~~~~~~~~~%

% Define if function to allow nested if statements in user-defined functions
function [val] = if__fun (cond, valT, valF)
% IF__FUN Select between two possible return values depending on the boolean
% variable COND.
    if (cond)
        val = valT;
    else
        val = valF;
    end
end

% initialize species function
function [species_init] = initialize_species( params )

    species_init = zeros(1,32);
    species_init(1) = params(3)*(params(1)+params(4));
    species_init(2) = params(5)*(params(2)+params(6));
    species_init(3) = params(32);
    species_init(4) = params(33);
    species_init(5) = params(34);
    species_init(6) = params(35);
    species_init(7) = params(36);
    species_init(8) = params(37);
    species_init(9) = 0;
    species_init(10) = 0;
    species_init(11) = 0;
    species_init(12) = 0;
    species_init(13) = 0;
    species_init(14) = 0;
    species_init(15) = 0;
    species_init(16) = 0;
    species_init(17) = 0;
    species_init(18) = 0;
    species_init(19) = 0;
    species_init(20) = 0;
    species_init(21) = 0;
    species_init(22) = 0;
    species_init(23) = 0;
    species_init(24) = 0;
    species_init(25) = 0;
    species_init(26) = 0;
    species_init(27) = 0;
    species_init(28) = 0;
    species_init(29) = 0;
    species_init(30) = 0;
    species_init(31) = 0;
    species_init(32) = 0;

end


% user-defined functions



% Calculate expressions
function [ expressions ] = calc_expressions ( parameters )

    expressions = zeros(1,39);
    expressions(1) = parameters(1);
    expressions(2) = parameters(2);
    expressions(3) = parameters(3);
    expressions(4) = parameters(4);
    expressions(5) = parameters(5);
    expressions(6) = parameters(6);
    expressions(7) = parameters(7);
    expressions(8) = parameters(8);
    expressions(9) = parameters(9);
    expressions(10) = parameters(10);
    expressions(11) = parameters(11);
    expressions(12) = parameters(12);
    expressions(13) = parameters(13);
    expressions(14) = parameters(14);
    expressions(15) = parameters(15);
    expressions(16) = parameters(16);
    expressions(17) = parameters(17);
    expressions(18) = parameters(18);
    expressions(19) = parameters(19);
    expressions(20) = parameters(20);
    expressions(21) = parameters(21);
    expressions(22) = parameters(22);
    expressions(23) = parameters(23);
    expressions(24) = parameters(24);
    expressions(25) = parameters(25);
    expressions(26) = parameters(26);
    expressions(27) = parameters(27);
    expressions(28) = parameters(28);
    expressions(29) = parameters(29);
    expressions(30) = parameters(30);
    expressions(31) = parameters(31);
    expressions(32) = parameters(32);
    expressions(33) = parameters(33);
    expressions(34) = parameters(34);
    expressions(35) = parameters(35);
    expressions(36) = parameters(36);
    expressions(37) = parameters(37);
    expressions(38) = (expressions(3)*(expressions(1)+expressions(4)));
    expressions(39) = (expressions(5)*(expressions(2)+expressions(6)));
   
end



% Calculate observables
function [ observables ] = calc_observables ( species, expressions )

    observables = zeros(1,12);
    observables(1) = species(2);
    observables(2) = species(1);
    observables(3) = species(11) +species(14) +species(25) +species(30);
    observables(4) = species(12) +species(13);
    observables(5) = species(15) +species(18) +species(19) +species(20) +species(21) +species(22) +species(23) +species(25) +2*species(26) +species(31);
    observables(6) = species(16) +species(21) +species(27);
    observables(7) = 2*species(17) +2*species(19) +2*species(24) +2*species(26) +2*species(31) +2*species(32);
    observables(8) = species(18) +species(20) +species(23);
    observables(9) = species(19) +species(21) +species(22) +species(24) +species(25) +2*species(26) +species(27) +species(28) +species(30) +2*species(31) +2*species(32);
    observables(10) = species(23) +species(29);
    observables(11) = species(5) +species(13) +species(15) +species(18) +species(19) +species(20) +species(21) +species(22) +species(23) +species(24) +species(25) +2*species(26) +species(27) +species(28) +species(30) +2*species(31) +2*species(32);
    observables(12) = species(7) +species(14) +species(16) +2*species(17) +2*species(19) +species(21) +species(22) +2*species(24) +species(25) +2*species(26) +species(27) +species(28) +species(30) +2*species(31) +2*species(32);

end


% Calculate ratelaws
function [ ratelaws ] = calc_ratelaws ( species, expressions, observables )

    ratelaws = zeros(1,12);
    ratelaws(1) = expressions(7)*species(2)*species(3);
    ratelaws(2) = expressions(11)*species(1)*species(4);
    ratelaws(3) = expressions(8)*species(9);
    ratelaws(4) = expressions(9)*species(9);
    ratelaws(5) = expressions(12)*species(10);
    ratelaws(6) = expressions(13)*species(10);
    ratelaws(7) = expressions(10)*species(11);
    ratelaws(8) = expressions(14)*species(12);
    ratelaws(9) = expressions(24)*species(5)*species(12);
    ratelaws(10) = expressions(18)*species(7)*species(11);
    ratelaws(11) = expressions(17)*species(13);
    ratelaws(12) = expressions(15)*species(13);
    ratelaws(13) = expressions(19)*species(14);
    ratelaws(14) = expressions(16)*species(15);
    ratelaws(15) = 0.5*expressions(20)*species(16)*species(16);
    ratelaws(16) = expressions(23)*species(16);
    ratelaws(17) = expressions(27)*species(15)*species(6);
    ratelaws(18) = expressions(22)*species(17);
    ratelaws(19) = expressions(21)*species(17);
    ratelaws(20) = 2*expressions(28)*species(15)*species(17);
    ratelaws(21) = expressions(29)*species(18)*species(8);
    ratelaws(22) = expressions(22)*species(19);
    ratelaws(23) = expressions(21)*species(19);
    ratelaws(24) = expressions(25)*species(19);
    ratelaws(25) = expressions(31)*species(20);
    ratelaws(26) = expressions(30)*species(20);
    ratelaws(27) = expressions(26)*species(19);
    ratelaws(28) = expressions(18)*species(22)*species(11);
    ratelaws(29) = expressions(20)*species(16)*species(21);
    ratelaws(30) = 0.5*expressions(20)*species(21)*species(21);
    ratelaws(31) = expressions(22)*species(24);
    ratelaws(32) = expressions(23)*species(21);
    ratelaws(33) = expressions(21)*species(24);
    ratelaws(34) = expressions(31)*species(23);
    ratelaws(35) = expressions(18)*species(28)*species(11);
    ratelaws(36) = expressions(19)*species(25);
    ratelaws(37) = expressions(20)*species(16)*species(27);
    ratelaws(38) = expressions(20)*species(21)*species(27);
    ratelaws(39) = 0.5*expressions(20)*species(27)*species(27);
    ratelaws(40) = expressions(22)*species(26);
    ratelaws(41) = expressions(23)*species(27);
    ratelaws(42) = expressions(21)*species(26);
    ratelaws(43) = expressions(19)*species(30);
    ratelaws(44) = expressions(22)*species(31);
    ratelaws(45) = expressions(22)*species(32);
    ratelaws(46) = expressions(21)*species(31);
    ratelaws(47) = expressions(21)*species(32);

end

% Calculate species derivatives
function [ Dspecies ] = calc_species_deriv ( time, species, expressions )
    
    % initialize derivative vector
    Dspecies = zeros(32,1);
    
    % update observables
    [ observables ] = calc_observables( species, expressions );
    
    % update ratelaws
    [ ratelaws ] = calc_ratelaws( species, expressions, observables );
                        
    % calculate derivatives
    Dspecies(1) = -ratelaws(2) +ratelaws(5) +ratelaws(8);
    Dspecies(2) = -ratelaws(1) +ratelaws(3) +ratelaws(7);
    Dspecies(3) = -ratelaws(1) +ratelaws(3) +ratelaws(7);
    Dspecies(4) = -ratelaws(2) +ratelaws(5) +ratelaws(8);
    Dspecies(5) = -ratelaws(9) +ratelaws(11) +ratelaws(14);
    Dspecies(6) = -ratelaws(17);
    Dspecies(7) = -ratelaws(10) +ratelaws(16) +2.0*ratelaws(19) +ratelaws(23) +ratelaws(33);
    Dspecies(8) = -ratelaws(21) +ratelaws(25);
    Dspecies(9) = ratelaws(1) -ratelaws(3) -ratelaws(4);
    Dspecies(10) = ratelaws(2) -ratelaws(5) -ratelaws(6);
    Dspecies(11) = ratelaws(4) -ratelaws(7) -ratelaws(10) +ratelaws(13) -ratelaws(28) -ratelaws(35) +ratelaws(36) +ratelaws(43);
    Dspecies(12) = ratelaws(6) -ratelaws(8) -ratelaws(9) +ratelaws(11) +ratelaws(12);
    Dspecies(13) = ratelaws(9) -ratelaws(11) -ratelaws(12);
    Dspecies(14) = ratelaws(10) -ratelaws(13);
    Dspecies(15) = ratelaws(12) -ratelaws(14) -ratelaws(17) -ratelaws(20) +ratelaws(24);
    Dspecies(16) = ratelaws(13) -2.0*ratelaws(15) -ratelaws(16) +2.0*ratelaws(18) +ratelaws(22) -ratelaws(29) +ratelaws(31) -ratelaws(37);
    Dspecies(17) = ratelaws(15) -ratelaws(18) -ratelaws(19) -ratelaws(20) +ratelaws(24);
    Dspecies(18) = ratelaws(17) -ratelaws(21) +ratelaws(25) +ratelaws(34);
    Dspecies(19) = ratelaws(20) -ratelaws(22) -ratelaws(23) -ratelaws(24) -ratelaws(27) +ratelaws(29);
    Dspecies(20) = ratelaws(21) -ratelaws(25) -ratelaws(26);
    Dspecies(21) = ratelaws(22) -ratelaws(29) -2.0*ratelaws(30) -ratelaws(32) +ratelaws(36) -ratelaws(38) +2.0*ratelaws(40) +ratelaws(44);
    Dspecies(22) = ratelaws(23) -ratelaws(28) +ratelaws(32) +2.0*ratelaws(42) +ratelaws(46);
    Dspecies(23) = ratelaws(26) -ratelaws(34);
    Dspecies(24) = ratelaws(27) -ratelaws(31) -ratelaws(33) +ratelaws(37);
    Dspecies(25) = ratelaws(28) -ratelaws(36);
    Dspecies(26) = ratelaws(30) -ratelaws(40) -ratelaws(42);
    Dspecies(27) = ratelaws(31) -ratelaws(37) -ratelaws(38) -2.0*ratelaws(39) -ratelaws(41) +ratelaws(43) +ratelaws(44) +2.0*ratelaws(45);
    Dspecies(28) = ratelaws(33) -ratelaws(35) +ratelaws(41) +ratelaws(46) +2.0*ratelaws(47);
    Dspecies(29) = ratelaws(34);
    Dspecies(30) = ratelaws(35) -ratelaws(43);
    Dspecies(31) = ratelaws(38) -ratelaws(44) -ratelaws(46);
    Dspecies(32) = ratelaws(39) -ratelaws(45) -ratelaws(47);

end


end
