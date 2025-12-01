/* extension.js
 *
 * T-FAN GNOME Shell Extension
 * Provides system tray integration, quick actions, and live training metrics
 */

import GObject from 'gi://GObject';
import St from 'gi://St';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Clutter from 'gi://Clutter';

import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';

const TFANIndicator = GObject.registerClass(
class TFANIndicator extends PanelMenu.Button {
    _init(extension) {
        super._init(0.0, 'T-FAN Indicator');

        this._extension = extension;
        this._settings = extension.getSettings();

        // Icon in top bar
        let icon = new St.Icon({
            gicon: Gio.icon_new_for_string(
                this._extension.path + '/assets/tfan-icon.svg'
            ),
            style_class: 'system-status-icon',
        });
        this.add_child(icon);

        // Training status label
        this._statusLabel = new St.Label({
            text: 'Idle',
            y_align: Clutter.ActorAlign.CENTER,
            style_class: 'tfan-status-label'
        });
        this.add_child(this._statusLabel);

        // Build menu
        this._buildMenu();

        // Start monitoring
        this._startMonitoring();
    }

    _buildMenu() {
        // Header with logo and status
        let headerBox = new St.BoxLayout({
            style_class: 'tfan-header-box',
            vertical: false
        });

        let logo = new St.Icon({
            gicon: Gio.icon_new_for_string(
                this._extension.path + '/assets/tfan-logo.svg'
            ),
            icon_size: 48
        });
        headerBox.add_child(logo);

        let titleBox = new St.BoxLayout({vertical: true});
        let title = new St.Label({
            text: 'T-FAN Neural Optimizer',
            style_class: 'tfan-title'
        });
        this._metricsLabel = new St.Label({
            text: 'No active training',
            style_class: 'tfan-metrics'
        });
        titleBox.add_child(title);
        titleBox.add_child(this._metricsLabel);
        headerBox.add_child(titleBox);

        let headerItem = new PopupMenu.PopupBaseMenuItem({reactive: false});
        headerItem.actor.add_child(headerBox);
        this.menu.addMenuItem(headerItem);

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        // Quick Actions
        let dashboardItem = new PopupMenu.PopupMenuItem('📊 Open Dashboard');
        dashboardItem.connect('activate', () => {
            this._launchApp('dashboard');
        });
        this.menu.addMenuItem(dashboardItem);

        let paretoItem = new PopupMenu.PopupMenuItem('🎯 Pareto Optimization');
        paretoItem.connect('activate', () => {
            this._launchApp('pareto');
        });
        this.menu.addMenuItem(paretoItem);

        let screensaverItem = new PopupMenu.PopupMenuItem('🌌 Topology Screensaver');
        screensaverItem.connect('activate', () => {
            this._launchApp('screensaver');
        });
        this.menu.addMenuItem(screensaverItem);

        let trainingItem = new PopupMenu.PopupMenuItem('🚀 Start Training');
        trainingItem.connect('activate', () => {
            this._launchTraining();
        });
        this.menu.addMenuItem(trainingItem);

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        // Live metrics submenu
        this._metricsSubmenu = new PopupMenu.PopupSubMenuMenuItem('📈 Live Metrics');
        this._accuracyItem = new PopupMenu.PopupMenuItem('Accuracy: --', {reactive: false});
        this._latencyItem = new PopupMenu.PopupMenuItem('Latency: --', {reactive: false});
        this._hvItem = new PopupMenu.PopupMenuItem('Hypervolume: --', {reactive: false});

        this._metricsSubmenu.menu.addMenuItem(this._accuracyItem);
        this._metricsSubmenu.menu.addMenuItem(this._latencyItem);
        this._metricsSubmenu.menu.addMenuItem(this._hvItem);
        this.menu.addMenuItem(this._metricsSubmenu);

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        // Settings
        let settingsItem = new PopupMenu.PopupMenuItem('⚙️ Settings');
        settingsItem.connect('activate', () => {
            this._extension.openPreferences();
        });
        this.menu.addMenuItem(settingsItem);
    }

    _launchApp(view = 'dashboard') {
        try {
            GLib.spawn_command_line_async(
                `tfan-gnome --view=${view}`
            );
        } catch (e) {
            logError(e, 'Failed to launch T-FAN app');
        }
    }

    _launchTraining() {
        try {
            GLib.spawn_command_line_async(
                'gnome-terminal -- bash -c "python training/train.py; exec bash"'
            );
        } catch (e) {
            logError(e, 'Failed to launch training');
        }
    }

    _startMonitoring() {
        // Monitor metrics file for updates
        this._metricsFile = Gio.File.new_for_path(
            GLib.get_home_dir() + '/.cache/tfan/metrics.json'
        );

        this._monitor = this._metricsFile.monitor_file(
            Gio.FileMonitorFlags.NONE,
            null
        );

        this._monitor.connect('changed', (monitor, file, otherFile, eventType) => {
            if (eventType === Gio.FileMonitorEvent.CHANGED ||
                eventType === Gio.FileMonitorEvent.CREATED) {
                this._updateMetrics();
            }
        });

        // Update every 5 seconds
        this._timeout = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, 5, () => {
            this._updateMetrics();
            return GLib.SOURCE_CONTINUE;
        });

        // Initial update
        this._updateMetrics();
    }

    _updateMetrics() {
        try {
            if (!this._metricsFile.query_exists(null)) return;

            let [success, contents] = this._metricsFile.load_contents(null);
            if (!success) return;

            let metrics = JSON.parse(new TextDecoder().decode(contents));

            // Update top bar status
            if (metrics.training_active) {
                this._statusLabel.text = `Step ${metrics.step || 0}`;
                this._statusLabel.style_class = 'tfan-status-active';
            } else {
                this._statusLabel.text = 'Idle';
                this._statusLabel.style_class = 'tfan-status-idle';
            }

            // Update menu metrics
            this._metricsLabel.text = metrics.training_active
                ? `Training: ${metrics.accuracy?.toFixed(3) || '--'} acc`
                : 'No active training';

            this._accuracyItem.label.text = `Accuracy: ${metrics.accuracy?.toFixed(3) || '--'}`;
            this._latencyItem.label.text = `Latency: ${metrics.latency_ms?.toFixed(1) || '--'} ms`;
            this._hvItem.label.text = `Hypervolume: ${metrics.hypervolume?.toFixed(0) || '--'}`;

        } catch (e) {
            // Silently fail if no metrics available
        }
    }

    destroy() {
        if (this._timeout) {
            GLib.source_remove(this._timeout);
            this._timeout = null;
        }

        if (this._monitor) {
            this._monitor.cancel();
            this._monitor = null;
        }

        super.destroy();
    }
});

export default class TFANExtension extends Extension {
    enable() {
        this._indicator = new TFANIndicator(this);
        Main.panel.addToStatusArea('tfan-indicator', this._indicator);
    }

    disable() {
        if (this._indicator) {
            this._indicator.destroy();
            this._indicator = null;
        }
    }
}
